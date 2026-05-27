# Mog Language Specification — Complete Reference for Code Generation

> Source: https://moglang.org/guide/spec, https://github.com/voltropy/mog
> Purpose: Training corpus for a code diffusion model that generates Mog programs.
> Mog is a statically typed, compiled, embedded language for AI agents. Full spec ~3200 tokens.

---

## 1. OVERVIEW & DESIGN PHILOSOPHY

Mog is a small, statically-typed, embeddable language for LLM agent scripting and plugin
development. It compiles to native code via rqbe (safe Rust QBE backend). Think of it as
a statically-typed Lua with async I/O and a capability-based security model.

Design Principles:
1. Small surface area — entire spec fits in LLM context window
2. Predictable semantics — no implicit coercion, no operator precedence
3. Familiar syntax — Rust/Go/TypeScript blend with Pythonisms
4. Safe by default — GC, bounds-checked, no null, no raw pointers
5. Host provides I/O — capability-based sandbox
6. Host provides compute — tensors exist, ML ops via host

NOT in Mog: classes, inheritance, generics (except Result<T>/?T/tensor<T>), macros,
exceptions, threads, raw pointers, syscalls, operator overloading, manual memory mgmt.

---

## 2. PROGRAM STRUCTURE

- Top-level: only declarations (functions, structs, capabilities, imports)
- Entry point: `fn main() -> int` (0 = success)
- Semicolons required
- Comments: `// single line` and `/* multi-line */`

```mog
fn main() -> int {
  println("Hello, world!");
  return 0;
}
```

---

## 3. VARIABLES & BINDINGS

- Declare with `:=` (type inferred), reassign with `=`
- Explicit type: `x: int = 42;`
- All variables mutable
- Shadowing via `:=` in same scope creates new binding
- No uninitialized variables

```mog
x := 42;               // initial binding (type inferred)
x = 100;               // reassignment
name: string = "hello"; // explicit type annotation
nums: [int] = [1, 2, 3];
scores: {string: int} = {};
```

---

## 4. TYPE SYSTEM

### Scalar Types
- `int` (i64 default), `float` (f64 default), `bool`, `string` (UTF-8, immutable, GC)

### Precision Types
- Signed integers: `i8`, `i16`, `i32`, `i64`
- Unsigned integers: `u8`, `u16`, `u32`, `u64`
- Floats: `f16`, `bf16`, `f32`, `f64`

### Composite Types
- Arrays: `[T]` — dynamic, homogeneous, GC-managed
- Maps: `{K: V}` — keys: int/float/string/bool
- Structs: named fields, no methods, no inheritance
- SoA: `soa StructName[N]` — columnar layout
- Tensors: `tensor<dtype>` — N-dimensional, fixed dtype
- Optional: `?T` — `some(value)` or `none`
- Result: `Result<T>` — `ok(value)` or `err(message)`
- Function types: `fn(T) -> U`

### Type Aliases
```mog
type Batch = tensor<f32>;
type Callback = fn(int) -> bool;
type Predicate = fn(int) -> bool;
```

### Conversions (explicit only)
- `as` for numeric casts (narrowing truncates silently)
- `str()` for string conversion
- `int_from_string()`, `parse_float()` return Result

```mog
x: i32 = 42;
y: i64 = x as i64;
s := str(42);
result := int_from_string("42");
```

---

## 5. OPERATORS (FLAT — NO PRECEDENCE)

ALL binary operators are flat. Mixed operators require parentheses.

### Associative (can chain with themselves):
`+`, `*`, `and`/`&&`, `or`/`||`, `&`, `|`

### Non-associative (cannot chain):
`-`, `/`, `%`, `==`, `!=`, `<`, `>`, `<=`, `>=`, `<<`, `>>`, `^`

### Unary (bind tighter than binary):
`-x`, `!x`

### Other:
`**` (power), `?` (error propagation), `..` (range), `as` (cast)

```mog
result := a + (b * c);              // OK — parens make grouping explicit
result := a + b * c;                // COMPILE ERROR — mixed + and *

check := (x > 0) && (y > 0);       // OK
check := x > 0 && y > 0;           // COMPILE ERROR — mixed > and &&

diff := (a - b) - c;               // OK — parenthesized
diff := a - b - c;                  // COMPILE ERROR — - is non-associative

a == b == c                         // PARSE ERROR: == is non-associative
```

String concatenation: `"hello" + " " + "world"`

---

## 6. CONTROL FLOW

### if/else (statement or expression)
```mog
if x > 0 {
  println("positive");
} else {
  println("non-positive");
}

val := if (x > 0) { 1 } else { 0 };

result := match x {
  0 => "zero",
  1 => "one",
  _ => "other"
};
```

### Loops
```mog
// while
while running {
  process();
}

// for..to (inclusive)
for i := 1 to 10 {
  println_i64(i);
}

// for..in range (exclusive upper bound)
for i in 0..10 {
  print(i);
}

// for over array
for item in items {
  println(item);
}

// for with index
for i, item in items {
  println(f"{i}: {item}");
}

// for over map
for key, value in config {
  println(f"{key}: {value}");
}
```

### break/continue
```mog
for i in 0..100 {
  if (i % 2) == 0 { continue; }
  if i > 50 { break; }
  println_i64(i);
}
```

### match (Result and Optional patterns)
```mog
match result {
  ok(n) => println(f"parsed: {n}"),
  err(msg) => println(f"failed: {msg}"),
}

match maybe_value {
  some(v) => println(f"got: {v}"),
  none => println("nothing"),
}

result: string = match n {
  0 => "zero",
  1 => "one",
  2 => "two",
  _ => "other"
};
```

---

## 7. FUNCTIONS

```mog
fn add(a: int, b: int) -> int {
  return a + b;
}

// Void function (no return type)
fn greet(name: string) {
  println(f"hello {name}");
}

// Named args with defaults
fn train(model: Model, data: Data, epochs: int = 10, lr: float = 0.001) -> Model {
  // ...
}
trained := train(model, data, epochs: 50, lr: 0.0001);

// Recursion
fn factorial(n: i64) -> i64 {
  if (n <= 1) { return 1; }
  return n * factorial(n - 1);
}

fn fibonacci(n: i64) -> i64 {
  if (n <= 0) { return 0; }
  if (n == 1) { return 1; }
  a := 0;
  b := 1;
  for i := 2 to n {
    tmp := b;
    b = a + b;
    a = tmp;
  }
  return b;
}
```

### Built-in Math Functions
`abs`, `sqrt`, `pow`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan2`, `exp`, `log`,
`log2`, `floor`, `ceil`, `round`, `min`, `max`

Constants: `PI`, `E`

### Built-in Print/Conversion
`println(s)`, `print_string(s)`, `print(n)`, `println_i64(n)`, `print_f64(x)`,
`str()`, `len()`, `int_from_string()`, `parse_float()`

---

## 8. CLOSURES & HIGHER-ORDER FUNCTIONS

```mog
// Anonymous function
double := fn(x: int) -> int { x * 2 };

// Closure capturing environment
fn make_adder(x: int) -> fn(int) -> int {
  return fn(y: int) -> int { x + y };
}

add5 := make_adder(5);
result := add5(3);  // 8

// Higher-order function
fn apply(f: fn(int) -> int, val: int) -> int {
  return f(val);
}

// Array methods with closures
evens := nums.filter(fn(x: int) -> bool { (x % 2) == 0 });
doubled := nums.map(fn(x: int) -> int { x * 2 });
sorted := nums.sort(fn(a: int, b: int) -> int { a - b });
```

Capture is by value (snapshot at closure creation time).

---

## 9. STRINGS

- Immutable UTF-8, double quotes only
- Escape sequences: `\n`, `\t`, `\\`, `\"`
- F-strings: `f"name={name}, age={age}"` — escape `{` with `{{`

```mog
greeting := "hello";
msg := f"result is {x + y}";

// Properties & methods
s.len
s.upper()
s.lower()
s.trim()
s.split(",")
s.contains("target")
s.starts_with("pre")
s.ends_with("suf")
s.replace("old", "new")
s[0:5]  // slicing
```

---

## 10. ARRAYS

```mog
numbers := [1, 2, 3];
empty: [int] = [];
zeros := [0; 100];           // repeat fill

numbers[0] = 10;             // mutation
numbers.push(4);
last := numbers.pop();
length := numbers.len;
slice := numbers[1:3];
joined := parts.join(", ");

// Higher-order
filtered := numbers.filter(fn(x: int) -> bool { x > 2 });
mapped := numbers.map(fn(x: int) -> int { x * 2 });
sorted := numbers.sort(fn(a: int, b: int) -> int { a - b });
```

---

## 11. MAPS

```mog
ages := {"alice": 30, "bob": 25};
ages["charlie"] = 35;

if ages.has("alice") {
  println_i64(ages["alice"]);
}

for key, value in ages {
  println(f"{key}: {value}");
}

keys := ages.keys();
vals := ages.values();
count := ages.len;
```

---

## 12. STRUCTS

```mog
struct Point { x: float, y: float }
struct Particle { x: f64, y: f64, mass: f64 }

p := Point { x: 1.0, y: 2.0 };
println(f"({p.x}, {p.y})");
p.x = 3.0;  // mutation

fn distance_sq(a: Point, b: Point) -> float {
  dx := a.x - b.x;
  dy := a.y - b.y;
  return (dx * dx) + (dy * dy);
}
```

Structs are heap-allocated, passed by reference. No methods, no inheritance.

### SoA (Struct of Arrays)
```mog
struct Datum { id: i64, val: i64 }

datums := soa Datum[100];
datums[0].id = 1;
datums[0].val = 42;
// Stored as per-field arrays: id: [i64; 100], val: [i64; 100]
```

---

## 13. ERROR HANDLING

### Result Type
```mog
fn safe_divide(a: int, b: int) -> Result {
  if b == 0 { return err("division by zero"); }
  return ok(a / b);
}

// ? propagation (caller must also return Result)
fn calc() -> Result {
  x := safe_divide(10, 2)?;  // unwraps or propagates error
  return ok(x + 1);
}

// Pattern matching
match safe_divide(10, 0) {
  ok(n) => println(f"result: {n}"),
  err(msg) => println(f"error: {msg}"),
}

// try/catch
try {
  data := fs.read("data.csv")?;
  process(data);
} catch(e) {
  println(f"failed: {e}");
}
```

### Optional Type
```mog
fn find_positive(n: int) -> ?int {
  if n > 0 { return some(n); }
  return none;
}

match find_positive(42) {
  some(v) => println(f"found: {v}"),
  none => println("not found"),
}

if result is some(idx) {
  println_i64(idx);
}
```

---

## 14. ASYNC/AWAIT

```mog
async fn fetch(url: string) -> Result<string> {
  response := await http.get(url)?;
  return ok(response.body);
}

async fn add_then_double(a: i64, b: i64) -> i64 {
  sum: i64 = await async_add(a, b);
  result: i64 = await async_double(sum);
  return result;
}

// Parallel execution
await all([task_alpha(), task_beta(), task_gamma()]);

// Race (first to finish wins)
winner: i64 = await race([fast_path(), slow_path()]);

// Fire and forget
spawn run_timer(1, 5000);

// Async main
async fn main() -> int {
  result := await fetch("https://api.example.com/data");
  return 0;
}
```

---

## 15. CAPABILITIES (SECURITY MODEL)

No built-in I/O. Host provides capabilities. Scripts declare what they need.

```mog
requires fs, http, model;  // must be provided by host
optional log, env;          // script works without them
```

### Standard Capabilities

**fs:**
- `read_file(path: string) -> string`
- `write_file(path: string, contents: string) -> int`
- `append_file(path: string, contents: string) -> int`
- `exists(path: string) -> bool`
- `remove(path: string) -> int`
- `file_size(path: string) -> int`

**process:**
- `sleep(ms: int) -> int` (async)
- `timestamp() -> int`
- `cwd() -> string`
- `getenv(name: string) -> string`
- `exit(code: int) -> int`

**log:**
- `info(msg: string)`, `warn(msg: string)`, `error(msg: string)`, `debug(msg: string)`

**http:**
- `get(url: string) -> Result` (async)
- `post(url: string, body: string) -> Result` (async)

### Custom Capabilities (.mogdecl)
```
capability env {
  fn get_name() -> string
  async fn delay_square(value: int, delay_ms: int) -> int
}

capability my_service {
  fn compute(x: int) -> int;
  async fn fetch(key: string) -> string;
}
```

### Capability Validation
- Compiler enforces: calling `fs.read_file()` requires `requires fs`
- Host enforces: missing required capability → script rejected at load

---

## 16. MODULES

Go-style module system:
```mog
// mog.mod declares module root
package mypackage;
import "path/to/module";
pub fn exported_function() { }
```

- `pub` for exports
- Circular import detection
- Single-file mode (no `package`) still supported

---

## 17. TENSORS

```mog
t := tensor<f16>([1.0, 2.0, 3.0]);
zeros := tensor<f32>.zeros([3, 224, 224]);
ones := tensor<f64>.ones([2, 3]);

s := t.shape;   // [3]
d := t.dtype;   // f16
n := t.ndim;    // 1
v := t[0];      // element access
reshaped := t.reshape([1, 3]);
```

All ML operations (matmul, etc.) via host `ml` capability.

---

## 18. PLUGINS

Compile to shared libraries, loaded by host via dlopen:
```bash
mogc --plugin math_plugin.mog -o math_plugin.dylib
```

Only `pub` functions are exported:
```mog
pub fn compute(x: int) -> int {
  return x * x;
}
```

---

## 19. COMPILATION

```bash
# Build compiler
cargo build --release --manifest-path compiler/Cargo.toml

# Compile to native binary
mogc program.mog -o program

# With optimization
mogc program.mog -o program -O1

# Emit QBE IR
mogc program.mog --emit-ir

# Compile as plugin
mogc program.mog --plugin mylib --plugin-version 1.0.0 -o mylib.dylib

# Link host capabilities
mogc program.mog --link host.rs -o program

# Convenience runner
./algb hello.mog
```

Pipeline: Lex → Parse → Type-check → QBE IR → rqbe → asm → native binary

---

## 20. EMBEDDING (HOST INTEGRATION)

### From Rust
```rust
use mog::compiler::{compile, compile_to_binary, CompileOptions};

let source = r#"fn main() { println("hello"); }"#;
let result = compile(source, None);
println!("{}", result.ir);  // QBE IL output
```

### From C
```c
#include "mog_compiler.h"

MogCompiler *c = mog_compiler_new();
MogCompileResult *r = mog_compile(c, source, source_len, NULL);
const char *ir = mog_result_ir(r);
```

---

## COMPLETE CODE EXAMPLES

### Example 1: Agent Hook (Post-Compaction)
```mog
import agent;
optional log;

pub fn on_post_compaction(session: agent.Session) {
  log.info("post-compaction hook: injecting reminder");

  session.messages.push(agent.Message {
    role: agent.Role.SYSTEM,
    content: "IMPORTANT: Always run tests before committing.",
  });
}
```

### Example 2: Async HTTP with Retry
```mog
async fn fetch_with_retry(url: string, max_retries: int) -> Result<string> {
  attempts := 0;
  for attempts < max_retries {
    match await http.get(url) {
      ok(response) => return ok(response.body),
      err(e) => {
        attempts = attempts + 1;
        if attempts >= max_retries {
          return err(f"failed after {max_retries} attempts: {e}");
        }
        println(f"attempt {attempts} failed, retrying...");
        await sleep(1000 * attempts);
      },
    }
  }
  return err(f"all {max_retries} attempts to fetch {url} failed");
}
```

### Example 3: Capability-Based Tool Hook
```mog
requires fs;
optional log;

pub fn on_tool_result(tool_name: string, stderr: string) {
  if (stderr.contains("permission denied")) {
    log.warn(f"{tool_name}: permission denied");
    fs.append_file("agent.log", f"[warn] {tool_name}: {stderr}\n");
  }
}
```

### Example 4: Host Capability Usage
```mog
requires fs, http, model;
optional log, env;

fn main() {
  data := fs.read("input.txt")?;
  result := await model.predict(data);
  fs.write("output.txt", result)?;
}
```

### Example 5: Guide Search (Full Program)
```mog
requires http, log;

struct SearchResult {
  title: string,
  url: string,
  score: int,
}

fn parse_results(response: string) -> [SearchResult] {
  results: [SearchResult] = [];
  r1 := SearchResult { title: "Intro to ML", url: "https://example.com/ml", score: 90 };
  results.push(r1);
  r2 := SearchResult { title: "Deep Learning", url: "https://example.com/dl", score: 85 };
  results.push(r2);
  r3 := SearchResult { title: "Unrelated Page", url: "https://example.com/other", score: 30 };
  results.push(r3);
  return results;
}

async fn search(query: string) -> Result<[SearchResult]> {
  response := await http.get(f"/api/search?q={query}");
  results := parse_results(response);
  log.info(f"found {results.len()} results for '{query}'");
  return ok(results);
}

async fn main() -> int {
  result := await search("machine learning");
  match result {
    ok(results) => {
      top := results.filter(fn(r: SearchResult) -> int { return r.score > 50; });
      for i, r in top {
        println(f"{r.title}: {r.url} (score: {r.score})");
      }
    },
    err(e) => {
      println(f"Search failed: {e}");
    },
  }
  return 0;
}
```

### Example 6: Async Timer with Spawn
```mog
requires process;

async fn run_timer(name: i64, ms: i64) -> i64 {
  println(f"[Timer {name}] Started: {ms}ms");
  await process.sleep(ms);
  println(f"[Timer {name}] DING! ({ms}ms elapsed)");
  return 0;
}

async fn main() -> i64 {
  spawn run_timer(1, 1000);
  spawn run_timer(2, 2000);
  spawn run_timer(3, 500);
  await process.sleep(3000);
  return 0;
}
```

### Example 7: Benchmark — Tiny (Fibonacci + Factorial)
```mog
fn fib(n: i64) -> i64 {
  if (n <= 1) { return n; }
  return fib(n - 1) + fib(n - 2);
}

fn factorial(n: i64) -> i64 {
  if (n <= 1) { return 1; }
  return n * factorial(n - 1);
}

fn main() -> i64 {
  println_i64(fib(10));
  println_i64(factorial(10));
  return 0;
}
```

### Example 8: Benchmark — Medium (Structs, Arrays, Closures, Match)
```mog
struct Point {
  x: i64,
  y: i64,
}

fn new_point(x: i64, y: i64) -> Point {
  return Point { x: x, y: y };
}

fn distance_sq(a: Point, b: Point) -> i64 {
  dx := a.x - b.x;
  dy := a.y - b.y;
  return (dx * dx) + (dy * dy);
}

fn build_squares(n: i64) -> [i64] {
  result := [0; n];
  for i in 0..n {
    result[i] = i * i;
  }
  return result;
}

fn filter_even(nums: [i64]) -> [i64] {
  result: [i64] = [];
  for v in nums {
    if ((v % 2) == 0) {
      result.push(v);
    }
  }
  return result;
}

fn sum_slice(nums: [i64]) -> i64 {
  total := 0;
  for v in nums {
    total = total + v;
  }
  return total;
}

fn bubble_sort(nums: [i64]) -> [i64] {
  n := nums.len;
  sorted := [0; n];
  for i in 0..n {
    sorted[i] = nums[i];
  }
  for i in 0..n {
    for j in 0..(n - i) - 1 {
      if (sorted[j] > sorted[j + 1]) {
        tmp := sorted[j];
        sorted[j] = sorted[j + 1];
        sorted[j + 1] = tmp;
      }
    }
  }
  return sorted;
}

fn classify(n: i64) -> string {
  result: string = match n {
    0 => "zero",
    1 => "one",
    2 => "two",
    3 => "three",
    _ => "other"
  };
  return result;
}

fn make_adder(x: i64) -> fn(i64) -> i64 {
  return fn(y: i64) -> i64 { x + y };
}

fn apply(f: fn(i64) -> i64, val: i64) -> i64 {
  return f(val);
}

fn format_point(p: Point) -> string {
  return f"({p.x}, {p.y})";
}

fn main() -> i64 {
  p1 := new_point(3, 4);
  p2 := new_point(0, 0);
  println(format_point(p1));
  println(format_point(p2));
  println(f"distance_sq: {distance_sq(p1, p2)}");

  nums := build_squares(20);
  evens := filter_even(nums);
  println(f"sum of squares: {sum_slice(nums)}");
  println(f"sum of even squares: {sum_slice(evens)}");

  unsorted: [i64] = [64, 34, 25, 12, 22, 11, 90];
  sorted := bubble_sort(unsorted);
  print_string("sorted: ");
  for i in 0..sorted.len {
    print_string(f"{sorted[i]} ");
  }
  println("");

  test_vals: [i64] = [0, 1, 2, 3, 42];
  for v in test_vals {
    println(f"{v} is {classify(v)}");
  }

  add5 := make_adder(5);
  add10 := make_adder(10);
  println(f"add5(3): {apply(add5, 3)}");
  println(f"add10(7): {apply(add10, 7)}");

  return 0;
}
```

### Example 9: Showcase Excerpts (Core Features)
```mog
fn add(a: i64, b: i64) -> i64 {
  return a + b;
}

struct Particle { x: f64, y: f64, mass: f64 }

fn safe_divide(a: i64, b: i64) -> Result {
  if b == 0 { return err("division by zero"); }
  return ok(a / b);
}

fn chain_divide(a: i64, b: i64, c: i64) -> Result {
  first := safe_divide(a, b)?;
  second := safe_divide(first, c)?;
  return ok(second);
}

fn find_positive(n: i64) -> ?i64 {
  if n > 0 { return some(n); }
  return none;
}

async fn add_then_double(a: i64, b: i64) -> i64 {
  sum: i64 = await async_add(a, b);
  result: i64 = await async_double(sum);
  return result;
}
```

### Example 10: Large Benchmark Excerpts (Number Theory, Closures, Matrix)
```mog
fn sqrt_approx(val: i64) -> i64 {
  if (val <= 0) { return 0; }
  guess := val / 2;
  for i in 0..30 {
    guess = (guess + (val / guess)) / 2;
  }
  return guess;
}

fn lcg_next(seed: i64) -> i64 {
  return ((seed * 1103515245) + 12345) % 2147483648;
}

fn make_multiplier(factor: i64) -> fn(i64) -> i64 {
  return fn(x: i64) -> i64 { x * factor };
}

fn is_prime(n: i64) -> bool {
  if n < 2 { return false; }
  if n < 4 { return true; }
  if (n % 2) == 0 { return false; }
  i := 3;
  while (i * i) <= n {
    if (n % i) == 0 { return false; }
    i = i + 2;
  }
  return true;
}

fn gcd(a: i64, b: i64) -> i64 {
  x := a;
  y := b;
  while y != 0 {
    tmp := y;
    y = x % y;
    x = tmp;
  }
  return x;
}

fn collatz_steps(n: i64) -> i64 {
  steps := 0;
  current := n;
  while current != 1 {
    if (current % 2) == 0 {
      current = current / 2;
    } else {
      current = (current * 3) + 1;
    }
    steps = steps + 1;
  }
  return steps;
}
```

---

## GRAMMAR/SYNTAX RULES SUMMARY

```
program       = (declaration)* EOF
declaration   = struct_decl | fn_decl | capability_decl | import_decl | type_alias
struct_decl   = "struct" IDENT "{" (field ("," field)*)? "}"
field         = IDENT ":" type
fn_decl       = ["pub"] ["async"] "fn" IDENT "(" params? ")" ["->" type] block
params        = param ("," param)*
param         = IDENT ":" type ["=" expr]
block         = "{" statement* "}"
type          = "int"|"i8"|"i16"|"i32"|"i64"|"u8"|"u16"|"u32"|"u64"
              | "float"|"f16"|"bf16"|"f32"|"f64"
              | "bool" | "string" | "?" type | "Result" ["<" type ">"]
              | "[" type "]" | "{" type ":" type "}"
              | "fn" "(" types? ")" ["->" type]
              | "tensor" "<" dtype ">"
              | IDENT

statement     = var_decl | assignment | expr_stmt | return_stmt | if_stmt
              | while_stmt | for_stmt | match_expr | break | continue
              | try_catch | spawn_stmt
var_decl      = IDENT ":=" expr ";"
              | IDENT ":" type "=" expr ";"
assignment    = lvalue "=" expr ";"
return_stmt   = "return" expr? ";"

if_stmt       = "if" ["("] expr [")"] block ["else" (if_stmt | block)]
while_stmt    = "while" expr block
for_stmt      = "for" IDENT "in" expr ".." expr block
              | "for" IDENT ":=" expr "to" expr block
              | "for" IDENT "in" expr block
              | "for" IDENT "," IDENT "in" expr block

match_expr    = "match" expr "{" (match_arm ("," match_arm)*)? "}"
match_arm     = pattern "=>" (expr | block)
pattern       = literal | IDENT | "ok" "(" IDENT ")" | "err" "(" IDENT ")"
              | "some" "(" IDENT ")" | "none" | "_"

expr          = literal | IDENT | expr binop expr | unop expr
              | expr "(" args? ")" | expr "[" expr "]" | expr "." IDENT
              | "fn" "(" params? ")" ["->" type] block
              | "await" expr | "if" expr block "else" block
              | expr "?" | "ok" "(" expr ")" | "err" "(" expr ")"
              | "some" "(" expr ")" | "none"
              | struct_literal | array_literal | map_literal
              | f_string | "soa" IDENT "[" expr "]"

capability    = "requires" IDENT ("," IDENT)* ";"
              | "optional" IDENT ("," IDENT)* ";"
import_decl   = "import" IDENT ";"
type_alias    = "type" IDENT "=" type ";"
spawn_stmt    = "spawn" expr ";"
try_catch     = "try" block "catch" "(" IDENT ")" block
```

---

## REPO STRUCTURE & TEST COVERAGE

GitHub: https://github.com/voltropy/mog (132 stars, MIT license)

### Key Files
- `showcase.mog` — 755-line complete feature demo
- `lang_spec.md` — formal language specification
- `docs/guide.md` — comprehensive language guide (17 chapters)
- `docs/context.md` — LLM context reference (~3200 tokens)
- `examples/` — guide_search.mog, timer.mog, plugins/, host integrations
- `benchmarks/mog/` — tiny.mog, medium.mog, large.mog
- `capabilities/*.mogdecl` — env, fs, http, log, math, process, timer

### Test Suite
- 1,139 passing tests across 15 test files
- Coverage: lexer, parser, analyzer, codegen, types, errors, features, security,
  memory limits, plugins, regressions, advanced (async, modules, tensors)

### Guide Chapters
1. First Program & Execution  2. Variables & Bindings  3. Types & Operators
4. Control Flow  5. Functions  6. Closures  7. Strings  8. Structs
9. Collections (Arrays & Maps)  10. Error Handling  11. Async/Await
12. Modules  13. Capabilities  14. Embedding  15. Plugins
16. Tensors  17. Advanced Topics & Cookbook
