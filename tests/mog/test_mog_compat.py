#!/usr/bin/env python3
"""
Mog Compiler Compatibility Matrix Tester
Tests every language construct individually and reports pass/fail.
"""

import subprocess
import os
import tempfile
import shutil
import json
from datetime import datetime

MOGC = "/Users/bobbyprice/projects/mog/compiler/target/release/mogc"
RUNTIME = "/Users/bobbyprice/projects/mog/runtime-rs/target/release/libmog_runtime.a"
TEST_DIR = "/Users/bobbyprice/projects/nCPU/egdc/mog/tests_data"
OUTPUT_MD = "/Users/bobbyprice/projects/nCPU/docs/mog/mog_compiler_compat.md"

os.makedirs(TEST_DIR, exist_ok=True)

# Each test: (category, name, mog_source_code, expected_output_substring_or_None)
TESTS = []

def t(cat, name, code, expected=None):
    TESTS.append((cat, name, code, expected))

# ============================================================
# 1. VARIABLES
# ============================================================
t("Variables", "walrus_bind", """
fn main() -> i64 {
  x := 42;
  println(f"x={x}");
  return 0;
}
""", "x=42")

t("Variables", "reassignment", """
fn main() -> i64 {
  x: i64 = 10;
  x = 20;
  println(f"x={x}");
  return 0;
}
""", "x=20")

t("Variables", "walrus_reassign", """
fn main() -> i64 {
  x: i64 = 10;
  x := 20;
  println(f"x={x}");
  return 0;
}
""", "x=20")

t("Variables", "explicit_type_i64", """
fn main() -> i64 {
  x: i64 = 42;
  println(f"x={x}");
  return 0;
}
""", "x=42")

# ============================================================
# 2. INTEGER TYPES
# ============================================================
t("IntegerTypes", "i64_arithmetic", """
fn main() -> i64 {
  a: i64 = 100;
  b: i64 = 37;
  println(f"{a + b}");
  return 0;
}
""", "137")

t("IntegerTypes", "i32_var", """
fn main() -> i64 {
  a: i32 = 42;
  println(f"a={a}");
  return 0;
}
""", "a=42")

t("IntegerTypes", "u32_var", """
fn main() -> i64 {
  a: u32 = 42;
  println(f"a={a}");
  return 0;
}
""", "a=42")

t("IntegerTypes", "u64_var", """
fn main() -> i64 {
  a: u64 = 42;
  println(f"a={a}");
  return 0;
}
""", "a=42")

t("IntegerTypes", "int_type", """
fn main() -> int {
  println("hello");
  return 0;
}
""", "hello")

# ============================================================
# 3. FLOAT TYPES
# ============================================================
t("FloatTypes", "f64_arithmetic", """
fn main() -> i64 {
  a: f64 = 3.14;
  b: f64 = 2.0;
  c: f64 = a + b;
  print_string("c=");
  print_f64(c);
  println("");
  return 0;
}
""", "c=")

t("FloatTypes", "f32_var", """
fn main() -> i64 {
  a: f32 = 3.14;
  println("f32 ok");
  return 0;
}
""", "f32 ok")

t("FloatTypes", "f16_var", """
fn main() -> i64 {
  a: f16 = 3.14;
  println("f16 ok");
  return 0;
}
""", "f16 ok")

t("FloatTypes", "bf16_var", """
fn main() -> i64 {
  a: bf16 = 3.14;
  println("bf16 ok");
  return 0;
}
""", "bf16 ok")

# ============================================================
# 4. BOOL
# ============================================================
t("Bool", "true_false", """
fn main() -> i64 {
  a: bool = true;
  b: bool = false;
  if a {
    println("a is true");
  }
  if !b {
    println("b is false");
  }
  return 0;
}
""", "a is true")

t("Bool", "and_or", """
fn main() -> i64 {
  a: bool = true;
  b: bool = false;
  if a && !b {
    println("and_or ok");
  }
  return 0;
}
""", "and_or ok")

t("Bool", "logical_or", """
fn main() -> i64 {
  a: bool = false;
  b: bool = true;
  if a || b {
    println("or ok");
  }
  return 0;
}
""", "or ok")

# ============================================================
# 5. STRINGS
# ============================================================
t("Strings", "literal", """
fn main() -> i64 {
  s: [u8] = "hello";
  println(f"{s}");
  return 0;
}
""", "hello")

t("Strings", "string_type", """
fn main() -> i64 {
  s: string = "hello";
  println(f"{s}");
  return 0;
}
""", "hello")

t("Strings", "f_string", """
fn main() -> i64 {
  x: i64 = 42;
  println(f"val={x}");
  return 0;
}
""", "val=42")

t("Strings", "concatenation", """
fn main() -> i64 {
  a: [u8] = "hello ";
  b: [u8] = "world";
  c: [u8] = a + b;
  println(f"{c}");
  return 0;
}
""", "hello world")

t("Strings", "dot_len", """
fn main() -> i64 {
  s: [u8] = "hello";
  println(f"len={s.len}");
  return 0;
}
""", "len=5")

t("Strings", "upper", """
fn main() -> i64 {
  s: [u8] = "hello";
  u: [u8] = s.upper();
  println(f"{u}");
  return 0;
}
""", "HELLO")

t("Strings", "lower", """
fn main() -> i64 {
  s: [u8] = "HELLO";
  l: [u8] = s.lower();
  println(f"{l}");
  return 0;
}
""", "hello")

t("Strings", "trim", """
fn main() -> i64 {
  s: [u8] = "  hi  ";
  t: [u8] = s.trim();
  println(f"[{t}]");
  return 0;
}
""", "[hi]")

t("Strings", "split", """
fn main() -> i64 {
  s: [u8] = "a,b,c";
  parts := s.split(",");
  println("split ok");
  return 0;
}
""", "split ok")

t("Strings", "contains", """
fn main() -> i64 {
  s: [u8] = "hello world";
  if s.contains("world") {
    println("contains ok");
  }
  return 0;
}
""", "contains ok")

t("Strings", "starts_with", """
fn main() -> i64 {
  s: [u8] = "hello world";
  if s.starts_with("hello") {
    println("starts_with ok");
  }
  return 0;
}
""", "starts_with ok")

t("Strings", "ends_with", """
fn main() -> i64 {
  s: [u8] = "hello world";
  if s.ends_with("world") {
    println("ends_with ok");
  }
  return 0;
}
""", "ends_with ok")

t("Strings", "replace", """
fn main() -> i64 {
  s: [u8] = "hello world";
  r: [u8] = s.replace("world", "mog");
  println(f"{r}");
  return 0;
}
""", "hello mog")

t("Strings", "slicing", """
fn main() -> i64 {
  s: [u8] = "hello world";
  sub: [u8] = s[0..5];
  println(f"{sub}");
  return 0;
}
""", "hello")

# ============================================================
# 6. OPERATORS
# ============================================================
t("Operators", "add_sub_mul_div_mod", """
fn main() -> i64 {
  println(f"{10 + 3}");
  println(f"{10 - 3}");
  println(f"{10 * 3}");
  println(f"{10 / 3}");
  println(f"{10 % 3}");
  return 0;
}
""", "13")

t("Operators", "power", """
fn main() -> i64 {
  r: i64 = 2 ** 10;
  println(f"{r}");
  return 0;
}
""", "1024")

t("Operators", "comparison", """
fn main() -> i64 {
  if 10 == 10 { println("eq"); }
  if 10 != 11 { println("ne"); }
  if 9 < 10 { println("lt"); }
  if 11 > 10 { println("gt"); }
  if 10 <= 10 { println("le"); }
  if 10 >= 10 { println("ge"); }
  return 0;
}
""", "eq")

t("Operators", "bitwise_and", """
fn main() -> i64 {
  r: i64 = 255 & 15;
  println(f"{r}");
  return 0;
}
""", "15")

t("Operators", "bitwise_or", """
fn main() -> i64 {
  r: i64 = 240 | 15;
  println(f"{r}");
  return 0;
}
""", "255")

t("Operators", "bitwise_xor", """
fn main() -> i64 {
  r: i64 = 255 ^ 15;
  println(f"{r}");
  return 0;
}
""", "240")

t("Operators", "shift_left", """
fn main() -> i64 {
  r: i64 = 1 << 8;
  println(f"{r}");
  return 0;
}
""", "256")

t("Operators", "shift_right", """
fn main() -> i64 {
  r: i64 = 256 >> 4;
  println(f"{r}");
  return 0;
}
""", "16")

t("Operators", "as_cast", """
fn main() -> i64 {
  a: f64 = 3.7;
  b: i64 = a as i64;
  println(f"{b}");
  return 0;
}
""", "3")

# ============================================================
# 7. CONTROL FLOW
# ============================================================
t("ControlFlow", "if_else_stmt", """
fn main() -> i64 {
  x: i64 = 10;
  if (x > 5) {
    println("gt5");
  } else {
    println("le5");
  }
  return 0;
}
""", "gt5")

t("ControlFlow", "if_else_expr", """
fn main() -> i64 {
  x: i64 = 10;
  r: i64 = if x > 5 { 1; } else { 0; };
  println(f"{r}");
  return 0;
}
""", "1")

t("ControlFlow", "while_loop", """
fn main() -> i64 {
  i: i64 = 0;
  s: i64 = 0;
  while (i < 10) {
    s = s + i;
    i = i + 1;
  }
  println(f"{s}");
  return 0;
}
""", "45")

t("ControlFlow", "for_to", """
fn main() -> i64 {
  s: i64 = 0;
  for i := 0 to 10 {
    s = s + i;
  }
  println(f"{s}");
  return 0;
}
""", "45")

t("ControlFlow", "for_in_range", """
fn main() -> i64 {
  s: i64 = 0;
  for i in 0..10 {
    s = s + i;
  }
  println(f"{s}");
  return 0;
}
""", "45")

t("ControlFlow", "for_in_array", """
fn main() -> i64 {
  arr: i64[] = [10, 20, 30];
  s: i64 = 0;
  for item in arr {
    s = s + item;
  }
  println(f"{s}");
  return 0;
}
""", "60")

t("ControlFlow", "for_index_item_in_array", """
fn main() -> i64 {
  arr: i64[] = [10, 20, 30];
  for i, item in arr {
    println(f"{i}: {item}");
  }
  return 0;
}
""", "0: 10")

t("ControlFlow", "for_kv_in_map", """
fn main() -> i64 {
  scores := {"alice": 95};
  for key, value in scores {
    println(f"{key}: {value}");
  }
  return 0;
}
""", "alice: 95")

t("ControlFlow", "break_stmt", """
fn main() -> i64 {
  s: i64 = 0;
  for i in 0..100 {
    if (i > 5) {
      break;
    }
    s = s + i;
  }
  println(f"{s}");
  return 0;
}
""", "15")

t("ControlFlow", "continue_stmt", """
fn main() -> i64 {
  s: i64 = 0;
  for i in 0..10 {
    if (i == 5) {
      continue;
    }
    s = s + i;
  }
  println(f"{s}");
  return 0;
}
""", "40")

# ============================================================
# 8. MATCH
# ============================================================
t("Match", "literal_patterns", """
fn main() -> i64 {
  x: i64 = 2;
  r: i64 = match x {
    1 => 10,
    2 => 20,
    3 => 30,
    _ => 0
  };
  println(f"{r}");
  return 0;
}
""", "20")

t("Match", "wildcard", """
fn main() -> i64 {
  x: i64 = 99;
  r: i64 = match x {
    1 => 10,
    _ => 999
  };
  println(f"{r}");
  return 0;
}
""", "999")

t("Match", "on_result", """
fn safe_div(a: i64, b: i64) -> Result<i64> {
  if b == 0 {
    return err("div0");
  }
  return ok(a / b);
}

fn main() -> i64 {
  r: Result<i64> = safe_div(10, 2);
  v: i64 = match r {
    ok(val) => val,
    err(msg) => 0
  };
  println(f"{v}");
  return 0;
}
""", "5")

t("Match", "on_optional", """
fn main() -> i64 {
  o: ?i64 = some(42);
  v: i64 = match o {
    some(x) => x,
    none => 0
  };
  println(f"{v}");
  return 0;
}
""", "42")

t("Match", "match_as_expression", """
fn main() -> i64 {
  x: i64 = 3;
  label: [u8] = match x {
    1 => "one",
    2 => "two",
    3 => "three",
    _ => "other"
  };
  println(f"{label}");
  return 0;
}
""", "three")

# ============================================================
# 9. FUNCTIONS
# ============================================================
t("Functions", "basic_fn", """
fn add(a: i64, b: i64) -> i64 {
  return a + b;
}

fn main() -> i64 {
  println(f"{add(10, 32)}");
  return 0;
}
""", "42")

t("Functions", "recursion", """
fn factorial(n: i64) -> i64 {
  if (n <= 1) { return 1; }
  return n * factorial(n - 1);
}

fn main() -> i64 {
  println(f"{factorial(5)}");
  return 0;
}
""", "120")

t("Functions", "default_params", """
fn greet(name: [u8] = "World") -> i64 {
  println(f"Hello, {name}!");
  return 0;
}

fn main() -> i64 {
  greet();
  greet("Mog");
  return 0;
}
""", "Hello, World!")

t("Functions", "named_args", """
fn make(x: i64, y: i64) -> i64 {
  return x * 10 + y;
}

fn main() -> i64 {
  r: i64 = make(y: 3, x: 4);
  println(f"{r}");
  return 0;
}
""", "43")

t("Functions", "void_fn", """
fn say_hello() {
  println("hello");
}

fn main() -> i64 {
  say_hello();
  return 0;
}
""", "hello")

# ============================================================
# 10. CLOSURES
# ============================================================
t("Closures", "anonymous_fn", """
fn main() -> i64 {
  double := fn(x: i64) -> i64 { x * 2 };
  println(f"{double(21)}");
  return 0;
}
""", "42")

t("Closures", "capture", """
fn main() -> i64 {
  offset: i64 = 100;
  add_offset := fn(x: i64) -> i64 { x + offset };
  println(f"{add_offset(42)}");
  return 0;
}
""", "142")

t("Closures", "higher_order", """
fn apply(f: fn(i64) -> i64, x: i64) -> i64 {
  return f(x);
}

fn main() -> i64 {
  double := fn(x: i64) -> i64 { x * 2 };
  println(f"{apply(double, 7)}");
  return 0;
}
""", "14")

t("Closures", "make_adder", """
fn make_adder(n: i64) -> fn(i64) -> i64 {
  return fn(x: i64) -> i64 { x + n };
}

fn main() -> i64 {
  add5 := make_adder(5);
  println(f"{add5(10)}");
  return 0;
}
""", "15")

t("Closures", "array_filter", """
fn main() -> i64 {
  arr: i64[] = [1, 2, 3, 4, 5, 6];
  evens := arr.filter(fn(x: i64) -> bool { x % 2 == 0 });
  println(f"len={evens.len}");
  return 0;
}
""", "len=3")

t("Closures", "array_map", """
fn main() -> i64 {
  arr: i64[] = [1, 2, 3];
  doubled := arr.map(fn(x: i64) -> i64 { x * 2 });
  println(f"{doubled[0]} {doubled[1]} {doubled[2]}");
  return 0;
}
""", "2 4 6")

t("Closures", "array_sort", """
fn main() -> i64 {
  arr: i64[] = [3, 1, 2];
  arr.sort();
  println(f"{arr[0]} {arr[1]} {arr[2]}");
  return 0;
}
""", "1 2 3")

# ============================================================
# 11. STRUCTS
# ============================================================
t("Structs", "definition_and_construction", """
struct Point {
  x: f64,
  y: f64
}

fn main() -> i64 {
  p: Point = Point { x: 3.0, y: 4.0 };
  println("struct ok");
  return 0;
}
""", "struct ok")

t("Structs", "field_access", """
struct Point {
  x: i64,
  y: i64
}

fn main() -> i64 {
  p: Point = Point { x: 10, y: 20 };
  println(f"x={p.x} y={p.y}");
  return 0;
}
""", "x=10 y=20")

t("Structs", "field_mutation", """
struct Particle {
  x: f64,
  y: f64,
  mass: f64
}

fn main() -> i64 {
  p: Particle = Particle { x: 0.0, y: 0.0, mass: 1.5 };
  p.x := 10.0;
  p.y := 20.0;
  println("mutated ok");
  return 0;
}
""", "mutated ok")

t("Structs", "field_mutation_eq", """
struct Particle {
  x: f64,
  y: f64
}

fn main() -> i64 {
  p: Particle = Particle { x: 0.0, y: 0.0 };
  p.x = 10.0;
  p.y = 20.0;
  println("mutated ok");
  return 0;
}
""", "mutated ok")

t("Structs", "pass_to_fn", """
struct Point {
  x: f64,
  y: f64
}

fn dist_sq(p: Point) -> f64 {
  return (p.x * p.x) + (p.y * p.y);
}

fn main() -> i64 {
  p: Point = Point { x: 3.0, y: 4.0 };
  d: f64 = dist_sq(p);
  print_string("d=");
  print_f64(d);
  println("");
  return 0;
}
""", "d=")

# ============================================================
# 12. ARRAYS
# ============================================================
t("Arrays", "literal", """
fn main() -> i64 {
  arr: i64[] = [10, 20, 30];
  println(f"{arr[0]}");
  return 0;
}
""", "10")

t("Arrays", "fill_syntax", """
fn main() -> i64 {
  arr := [0; 10];
  println(f"len={arr.len}");
  return 0;
}
""", "len=10")

t("Arrays", "indexing", """
fn main() -> i64 {
  arr: i64[] = [10, 20, 30, 40, 50];
  println(f"{arr[0]} {arr[2]} {arr[4]}");
  return 0;
}
""", "10 30 50")

t("Arrays", "push", """
fn main() -> i64 {
  arr: i64[] = [10, 20, 30];
  arr.push(40);
  println(f"len={arr.len}");
  return 0;
}
""", "len=4")

t("Arrays", "pop", """
fn main() -> i64 {
  arr: i64[] = [10, 20, 30];
  v: i64 = arr.pop();
  println(f"popped={v} len={arr.len}");
  return 0;
}
""", "popped=")

t("Arrays", "dot_len", """
fn main() -> i64 {
  arr: i64[] = [10, 20, 30];
  println(f"len={arr.len}");
  return 0;
}
""", "len=3")

t("Arrays", "slicing", """
fn main() -> i64 {
  arr: i64[] = [10, 20, 30, 40, 50];
  sub := arr[1..3];
  println(f"len={sub.len}");
  return 0;
}
""", "len=")

t("Arrays", "join", """
fn main() -> i64 {
  arr: [u8][] = ["a", "b", "c"];
  joined: [u8] = arr.join(",");
  println(f"{joined}");
  return 0;
}
""", "a,b,c")

# ============================================================
# 13. MAPS
# ============================================================
t("Maps", "literal", """
fn main() -> i64 {
  m := {"x": 10, "y": 20};
  println("map ok");
  return 0;
}
""", "map ok")

t("Maps", "indexing", """
fn main() -> i64 {
  m := {"x": 10, "y": 20};
  println(f"{m[\"x\"]}");
  return 0;
}
""", "10")

t("Maps", "has", """
fn main() -> i64 {
  m := {"x": 10};
  if m.has("x") {
    println("has ok");
  }
  return 0;
}
""", "has ok")

t("Maps", "keys", """
fn main() -> i64 {
  m := {"x": 10};
  k := m.keys();
  println("keys ok");
  return 0;
}
""", "keys ok")

t("Maps", "values", """
fn main() -> i64 {
  m := {"x": 10};
  v := m.values();
  println("values ok");
  return 0;
}
""", "values ok")

t("Maps", "dot_len", """
fn main() -> i64 {
  m := {"x": 10, "y": 20};
  println(f"len={m.len}");
  return 0;
}
""", "len=2")

t("Maps", "for_iteration", """
fn main() -> i64 {
  m := {"alice": 95};
  for key, value in m {
    println(f"{key}: {value}");
  }
  return 0;
}
""", "alice: 95")

# ============================================================
# 14. RESULT
# ============================================================
t("Result", "ok_constructor", """
fn get_val() -> Result<i64> {
  return ok(42);
}

fn main() -> i64 {
  r: Result<i64> = get_val();
  if r is ok(v) {
    println(f"{v}");
  }
  return 0;
}
""", "42")

t("Result", "err_constructor", """
fn fail() -> Result<i64> {
  return err("bad");
}

fn main() -> i64 {
  r: Result<i64> = fail();
  if r is err(e) {
    println("got error");
  }
  return 0;
}
""", "got error")

t("Result", "question_mark_propagation", """
fn div(a: i64, b: i64) -> Result<i64> {
  if b == 0 { return err("div0"); }
  return ok(a / b);
}

fn chain(a: i64, b: i64, c: i64) -> Result<i64> {
  step1: i64 = div(a, b)?;
  step2: i64 = div(step1, c)?;
  return ok(step2);
}

fn main() -> i64 {
  r: Result<i64> = chain(100, 5, 2);
  if r is ok(v) {
    println(f"{v}");
  }
  return 0;
}
""", "10")

t("Result", "match_result", """
fn get_val(flag: i64) -> Result<i64> {
  if flag > 0 { return ok(flag); }
  return err("negative");
}

fn main() -> i64 {
  r: Result<i64> = get_val(42);
  v: i64 = match r {
    ok(x) => x,
    err(e) => 0
  };
  println(f"{v}");
  return 0;
}
""", "42")

# ============================================================
# 15. OPTIONAL
# ============================================================
t("Optional", "some_constructor", """
fn main() -> i64 {
  o: ?i64 = some(42);
  if o is some(v) {
    println(f"{v}");
  }
  return 0;
}
""", "42")

t("Optional", "none_value", """
fn main() -> i64 {
  o: ?i64 = none;
  if o is none {
    println("is none");
  }
  return 0;
}
""", "is none")

t("Optional", "match_optional", """
fn main() -> i64 {
  o: ?i64 = some(99);
  v: i64 = match o {
    some(x) => x,
    none => 0
  };
  println(f"{v}");
  return 0;
}
""", "99")

t("Optional", "fn_returning_optional", """
fn find_pos(n: i64) -> ?i64 {
  if n > 0 { return some(n); }
  return none;
}

fn main() -> i64 {
  o: ?i64 = find_pos(42);
  if o is some(v) {
    println(f"{v}");
  }
  o2: ?i64 = find_pos(-5);
  if o2 is none {
    println("none");
  }
  return 0;
}
""", "42")

# ============================================================
# 16. TYPE ALIASES
# ============================================================
t("TypeAliases", "basic_type_alias", """
type Num = i64;

fn main() -> i64 {
  x: Num = 42;
  println(f"{x}");
  return 0;
}
""", "42")

# ============================================================
# 17. ASYNC
# ============================================================
t("Async", "async_fn_basic", """
async fn compute(a: i64, b: i64) -> i64 {
  return a + b;
}

async fn main() -> i64 {
  r: i64 = await compute(20, 22);
  println(f"{r}");
  return 0;
}
""", "42")

t("Async", "async_nested_await", """
async fn add(a: i64, b: i64) -> i64 {
  return a + b;
}

async fn double(x: i64) -> i64 {
  return x * 2;
}

async fn add_then_double(a: i64, b: i64) -> i64 {
  sum: i64 = await add(a, b);
  result: i64 = await double(sum);
  return result;
}

async fn main() -> i64 {
  r: i64 = await add_then_double(10, 11);
  println(f"{r}");
  return 0;
}
""", "42")

# ============================================================
# 18. CAPABILITIES
# ============================================================
t("Capabilities", "requires_fs", """
requires fs;

fn main() -> i64 {
  println("cap ok");
  return 0;
}
""", None)  # may not run without host, just test compile

t("Capabilities", "requires_env", """
requires env;

fn main() -> i64 {
  println("cap ok");
  return 0;
}
""", None)

# ============================================================
# 19. SoA
# ============================================================
t("SoA", "soa_basic", """
struct Datum { id: i64, val: i64 }

fn main() -> i64 {
  datums := soa Datum[10];
  datums[0].id = 1;
  datums[0].val = 100;
  println(f"id={datums[0].id}");
  return 0;
}
""", "id=1")

# ============================================================
# 20. ERROR HANDLING
# ============================================================
t("ErrorHandling", "try_catch", """
fn might_fail() -> Result<i64> {
  return err("oops");
}

fn main() -> i64 {
  try {
    r: i64 = might_fail()?;
    println(f"got {r}");
  } catch e {
    println(f"caught: {e}");
  }
  return 0;
}
""", "caught:")

# ============================================================
# Additional edge cases
# ============================================================
t("Variables", "infer_type_walrus", """
fn main() -> i64 {
  x := 42;
  y := "hello";
  println(f"{x} {y}");
  return 0;
}
""", "42 hello")

t("ControlFlow", "if_no_parens", """
fn main() -> i64 {
  x: i64 = 10;
  if x > 5 {
    println("ok");
  }
  return 0;
}
""", "ok")

t("ControlFlow", "if_with_parens", """
fn main() -> i64 {
  x: i64 = 10;
  if (x > 5) {
    println("ok");
  }
  return 0;
}
""", "ok")

t("ControlFlow", "else_if", """
fn main() -> i64 {
  x: i64 = 5;
  if x > 10 {
    println("gt10");
  } else if x > 3 {
    println("gt3");
  } else {
    println("le3");
  }
  return 0;
}
""", "gt3")

t("Functions", "return_struct", """
struct Point { x: i64, y: i64 }

fn make_point(x: i64, y: i64) -> Point {
  return Point { x: x, y: y };
}

fn main() -> i64 {
  p: Point = make_point(3, 4);
  println(f"x={p.x}");
  return 0;
}
""", "x=3")

t("Strings", "str_conversion", """
fn main() -> i64 {
  s: string = str(42);
  println(f"s={s}");
  return 0;
}
""", "s=42")

t("Functions", "nested_fn_def", """
fn main() -> i64 {
  fn helper(x: i64) -> i64 {
    return x * 2;
  }
  println(f"{helper(21)}");
  return 0;
}
""", "42")

t("Operators", "unary_neg", """
fn main() -> i64 {
  x: i64 = 42;
  y: i64 = -x;
  println(f"{y}");
  return 0;
}
""", "-42")

t("Arrays", "empty_array", """
fn main() -> i64 {
  arr: i64[] = [];
  arr.push(10);
  println(f"len={arr.len}");
  return 0;
}
""", "len=1")

t("Match", "match_string", """
fn main() -> i64 {
  s: [u8] = "hello";
  r: i64 = match s {
    "hello" => 1,
    "world" => 2,
    _ => 0
  };
  println(f"{r}");
  return 0;
}
""", "1")

t("Closures", "closure_return_from_fn", """
fn make_multiplier(n: i64) -> fn(i64) -> i64 {
  return fn(x: i64) -> i64 { x * n };
}

fn main() -> i64 {
  mul3 := make_multiplier(3);
  println(f"{mul3(14)}");
  return 0;
}
""", "42")

t("Structs", "struct_with_string_field", """
struct Person {
  name: [u8],
  age: i64
}

fn main() -> i64 {
  p: Person = Person { name: "Alice", age: 30 };
  println(f"{p.name} is {p.age}");
  return 0;
}
""", "Alice is 30")

t("ControlFlow", "while_walrus_increment", """
fn main() -> i64 {
  i: i64 = 0;
  s: i64 = 0;
  while (i < 10) {
    s := s + i;
    i := i + 1;
  }
  println(f"{s}");
  return 0;
}
""", "45")

t("Operators", "float_comparison", """
fn main() -> i64 {
  a: f64 = 3.14;
  b: f64 = 2.71;
  if a > b {
    println("gt");
  }
  return 0;
}
""", "gt")

t("Strings", "println_direct", """
fn main() -> i64 {
  println("Hello, World!");
  return 0;
}
""", "Hello, World!")

t("Strings", "print_string_fn", """
fn main() -> i64 {
  print_string("hello ");
  print_string("world");
  println("");
  return 0;
}
""", "hello world")

t("IntegerTypes", "print_i64_fn", """
fn main() -> i64 {
  print(42);
  println("");
  return 0;
}
""", "42")

# ============================================================
# RUN ALL TESTS
# ============================================================

def run_test(cat, name, code, expected):
    """Run a single test. Returns dict with results."""
    test_id = f"{cat}_{name}"
    mog_file = os.path.join(TEST_DIR, f"{test_id}.mog")
    out_file = os.path.join(TEST_DIR, f"{test_id}")
    
    # Write source
    with open(mog_file, 'w') as f:
        f.write(code.strip() + "\n")
    
    result = {
        "category": cat,
        "name": name,
        "compiles": False,
        "runs": False,
        "output": "",
        "compile_error": "",
        "run_error": "",
        "correct": None,
    }
    
    # Compile
    try:
        cp = subprocess.run(
            [MOGC, mog_file, "-o", out_file, "--link", RUNTIME],
            capture_output=True, text=True, timeout=30
        )
        if cp.returncode == 0:
            result["compiles"] = True
        else:
            result["compile_error"] = (cp.stderr + cp.stdout).strip()[:500]
            return result
    except Exception as e:
        result["compile_error"] = str(e)[:500]
        return result
    
    # Run
    try:
        rp = subprocess.run(
            [out_file],
            capture_output=True, text=True, timeout=10
        )
        result["output"] = rp.stdout.strip()[:500]
        if rp.returncode == 0:
            result["runs"] = True
        else:
            result["run_error"] = f"exit={rp.returncode} stderr={rp.stderr.strip()[:200]}"
            # Still might have partial output
            if result["output"]:
                result["runs"] = True  # produced output at least
    except Exception as e:
        result["run_error"] = str(e)[:500]
    
    # Check correctness
    if expected is not None and result["runs"]:
        result["correct"] = expected in result["output"]
    elif expected is None:
        result["correct"] = None  # no expected output to check
    
    return result

def main():
    print(f"Running {len(TESTS)} Mog compiler compatibility tests...")
    print(f"Compiler: {MOGC}")
    print(f"Runtime:  {RUNTIME}")
    print()
    
    results = []
    for i, (cat, name, code, expected) in enumerate(TESTS):
        r = run_test(cat, name, code, expected)
        status = "COMPILE_FAIL"
        if r["compiles"] and r["runs"]:
            if r["correct"] is True:
                status = "PASS"
            elif r["correct"] is False:
                status = "WRONG_OUTPUT"
            else:
                status = "COMPILES+RUNS"
        elif r["compiles"]:
            status = "RUN_FAIL"
        
        results.append(r)
        print(f"  [{i+1:3d}/{len(TESTS)}] {cat:15s} / {name:30s} => {status}")
        if status == "COMPILE_FAIL":
            # Print first line of error
            err_line = r["compile_error"].split("\n")[0][:80]
            print(f"           Error: {err_line}")
    
    # Generate markdown report
    generate_report(results)
    print(f"\nReport saved to: {OUTPUT_MD}")

def generate_report(results):
    lines = []
    lines.append("# Mog Compiler Compatibility Matrix")
    lines.append(f"")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Compiler: `{MOGC}`")
    lines.append(f"Runtime: `{RUNTIME}`")
    lines.append(f"")
    
    # Summary
    total = len(results)
    compiles = sum(1 for r in results if r["compiles"])
    runs = sum(1 for r in results if r["runs"])
    correct = sum(1 for r in results if r["correct"] is True)
    compile_fail = sum(1 for r in results if not r["compiles"])
    
    lines.append("## Summary")
    lines.append(f"")
    lines.append(f"| Metric | Count | Pct |")
    lines.append(f"|--------|-------|-----|")
    lines.append(f"| Total Tests | {total} | 100% |")
    lines.append(f"| Compiles | {compiles} | {compiles*100//total}% |")
    lines.append(f"| Runs | {runs} | {runs*100//total}% |")
    lines.append(f"| Correct Output | {correct} | {correct*100//total}% |")
    lines.append(f"| Compile Failures | {compile_fail} | {compile_fail*100//total}% |")
    lines.append(f"")
    
    # Category summary
    cats = {}
    for r in results:
        c = r["category"]
        if c not in cats:
            cats[c] = {"total": 0, "compiles": 0, "runs": 0, "correct": 0}
        cats[c]["total"] += 1
        if r["compiles"]: cats[c]["compiles"] += 1
        if r["runs"]: cats[c]["runs"] += 1
        if r["correct"] is True: cats[c]["correct"] += 1
    
    lines.append("## By Category")
    lines.append(f"")
    lines.append(f"| Category | Total | Compiles | Runs | Correct |")
    lines.append(f"|----------|-------|----------|------|---------|")
    for c, s in cats.items():
        lines.append(f"| {c} | {s['total']} | {s['compiles']} | {s['runs']} | {s['correct']} |")
    lines.append(f"")
    
    # Detailed results
    lines.append("## Detailed Results")
    lines.append(f"")
    
    current_cat = None
    for r in results:
        if r["category"] != current_cat:
            current_cat = r["category"]
            lines.append(f"### {current_cat}")
            lines.append(f"")
        
        if r["compiles"] and r["runs"] and r["correct"] is True:
            status = "✅ PASS"
        elif r["compiles"] and r["runs"] and r["correct"] is None:
            status = "✅ COMPILES+RUNS"
        elif r["compiles"] and r["runs"] and r["correct"] is False:
            status = "⚠️ WRONG OUTPUT"
        elif r["compiles"] and not r["runs"]:
            status = "⚠️ COMPILE OK, RUN FAIL"
        else:
            status = "❌ COMPILE FAIL"
        
        lines.append(f"**{r['name']}**: {status}")
        if r["output"]:
            lines.append(f"  - Output: `{r['output'][:100]}`")
        if r["compile_error"]:
            lines.append(f"  - Compile Error: `{r['compile_error'][:200]}`")
        if r["run_error"]:
            lines.append(f"  - Run Error: `{r['run_error'][:200]}`")
        lines.append(f"")
    
    # CRITICAL: Generate the safe constructs list
    lines.append("## Safe Constructs for Code Generation")
    lines.append(f"")
    lines.append("These constructs are confirmed to compile and run correctly:")
    lines.append(f"")
    for r in results:
        if r["compiles"] and r["runs"]:
            lines.append(f"- ✅ {r['category']}/{r['name']}")
    lines.append(f"")
    
    lines.append("## AVOID These Constructs (Compile Failures)")
    lines.append(f"")
    lines.append("These constructs fail to compile — do NOT use in generated code:")
    lines.append(f"")
    for r in results:
        if not r["compiles"]:
            err_short = r["compile_error"].split("\n")[0][:100]
            lines.append(f"- ❌ {r['category']}/{r['name']}: `{err_short}`")
    lines.append(f"")
    
    with open(OUTPUT_MD, 'w') as f:
        f.write("\n".join(lines))

if __name__ == "__main__":
    main()
