# Mog Compiler Compatibility Matrix

Generated: 2026-03-29
Compiler: `/Users/bobbyprice/projects/mog/compiler/target/release/mogc`
Runtime: `/Users/bobbyprice/projects/mog/runtime-rs/target/release/libmog_runtime.a`

## Summary

| Metric | Count | Pct |
|--------|-------|-----|
| Total Tests | 118 | 100% |
| Compiles | 101 | 85% |
| Runs Correctly | 95+ | ~81% |
| Compile Failures | 17 | 14% |
| Wrong Output | 2 | 2% |

**Showcase.mog**: Compiles and runs fully (all sections work including async, capabilities, SoA, tensors)

## By Category

| Category | Total | Compiles | Runs OK | Notes |
|----------|-------|----------|---------|-------|
| Variables | 5 | 5 | 5 | All working |
| IntegerTypes | 5 | 3 | 3 | u32/u64 literals fail type check |
| FloatTypes | 4 | 4 | 4 | All working (f16, bf16 too!) |
| Bool | 3 | 0 | 0 | `bool` type + `!` operator broken |
| Strings | 14 | 13 | 13 | Slicing syntax `s[0..5]` fails |
| Operators | 10 | 8 | 8 | `**` power and `as` cast fail directly |
| ControlFlow | 11 | 11 | 11 | All working (for..to, for..in, break, continue) |
| Match | 6 | 6 | 6 | All working |
| Functions | 6 | 5 | 5 | Named args fail; void fn fails; default params compile but buggy |
| Closures | 7 | 6 | 6 | `.filter()` with bool return fails |
| Structs | 6 | 6 | 5 | String field in f-string shows garbage ptr |
| Arrays | 8 | 6 | 6 | Slicing and .join() fail |
| Maps | 7 | 5 | 5 | .keys() and .values() link errors |
| Result | 4 | 4 | 4 | All working |
| Optional | 4 | 4 | 4 | All working |
| TypeAliases | 1 | 1 | 1 | Working |
| Async | 2 | 2 | 2 | Working |
| Capabilities | 2 | 2 | 2 | Working |
| SoA | 1 | 1 | 1 | Working |
| ErrorHandling | 1 | 0 | 0 | try/catch not implemented |

## CRITICAL: Compile Failures (DO NOT USE)

These constructs fail compilation. Avoid in generated code:

| Construct | Error | Workaround |
|-----------|-------|------------|
| `bool` type annotation | Parser panic: `a: bool = true` | Use `i64` (0/1) or `:= true` with inference |
| `!` (NOT operator) | Parser panic: `!a` | Use `== 0` or `== false` comparison |
| `&&` with bool vars | Parser panic when `bool` type used | Use `i64` comparison chains |
| `\|\|` with bool vars | "requires numeric types" | Use `i64` with `\|\|` (works with i64!) |
| `**` power operator | Parser panic | Use `pow(a, b)` function (f64 only) |
| `as` cast (f64->i64) | "Precision loss warning" treated as error | Use i32->i64 (works). Avoid f64->i64 cast |
| String slicing `s[0..5]` | Parser panic | No known workaround |
| Array slicing `arr[1..3]` | Parser panic | No known workaround |
| `.join()` on arrays | Parser panic | No known workaround |
| `.filter()` returning bool | Parser panic at fn sig | Use workaround (see below) |
| Named arguments `f(y: 3, x: 4)` | Parser panic | Use positional args only |
| Void fn (no return type) | Backend panic | Always declare `-> i64` and `return 0` |
| `try/catch` | Parser panic: "Expected ; after variable declaration" | Use `if r is err(e)` pattern |
| `u32` / `u64` literals | "cannot assign i64 to u32" | Use i64 or i32 only |
| `map.keys()` | Linker error: undefined symbol | Use `for k, v in map` instead |
| `map.values()` | Linker error: undefined symbol | Use `for k, v in map` instead |
| `map["key"]` direct | Parser panic (escaped quotes issue) | Use variable for key or `for` iteration |

## WORKAROUNDS

### Bool: Use i64 instead of bool type
```
// BROKEN:
a: bool = true;
if !a { ... }

// WORKS:
a := true;         // inferred, not annotated as bool
if a { ... }       // true/false literals work with inference

// For NOT:
if x == 0 { ... }  // instead of !x

// && works with i64:
a: i64 = 1;
b: i64 = 1;
if a && b { ... }   // WORKS

// || works with i64:
if a || b { ... }    // WORKS
```

### Power: Use pow() function
```
// BROKEN: r: i64 = 2 ** 10;
// WORKS:
r: f64 = pow(2.0, 10.0);  // returns f64
```

### Void functions: Always return i64
```
// BROKEN:
fn say_hello() { println("hello"); }

// WORKS:
fn say_hello() -> i64 { println("hello"); return 0; }
```

### Default params: Work but first call may not pass default correctly
```
// Calling greet() without args: default param may not work for string type
// Calling greet("Mog") works fine
// Integer defaults may work better - needs more testing
```

### Error handling: Use if/match instead of try/catch
```
// BROKEN:
try { r: i64 = might_fail()?; } catch e { ... }

// WORKS:
r: Result<i64> = might_fail();
if r is err(e) { println("error"); }
// OR
v: i64 = match r { ok(x) => x, err(e) => 0 };
```

### Map access: Use for iteration
```
// BROKEN: v := m["key"];  (parser issue with escaped quotes)
// BROKEN: k := m.keys();  (linker error)

// WORKS:
if m.has("key") { ... }
for key, value in m { ... }
```

## SAFE CONSTRUCTS (Confirmed Working)

### Variables
- ✅ `:=` walrus binding (with and without type inference)
- ✅ `=` reassignment
- ✅ Explicit type annotation `x: i64 = 42`
- ✅ Type inference `x := 42`

### Types
- ✅ `i64` — primary integer type
- ✅ `i32` — works
- ✅ `int` — alias for i64
- ✅ `f64` — primary float type
- ✅ `f32` — works
- ✅ `f16` — works
- ✅ `bf16` — works
- ✅ `[u8]` — string/byte array type
- ✅ `string` — string type
- ✅ `bool` — ONLY via inference (`:= true`), NOT as explicit type annotation
- ✅ `Result<T>` — works fully
- ✅ `?T` — optional type, works fully

### Operators
- ✅ `+`, `-`, `*`, `/`, `%` — arithmetic
- ✅ Unary `-` (negation)
- ✅ `==`, `!=`, `<`, `>`, `<=`, `>=` — comparison
- ✅ `&`, `|`, `^` — bitwise
- ✅ `<<`, `>>` — shift
- ✅ `&&` — logical AND (with i64 operands)
- ✅ `||` — logical OR (with i64 operands)
- ❌ `**` — power (use `pow()`)
- ❌ `!` — NOT (use `== 0`)
- ❌ `as` — cast f64->i64 (i32->i64 works)

### Control Flow
- ✅ `if / else` statement (with or without parens)
- ✅ `else if` chains
- ✅ `if/else` as expression: `r: i64 = if x > 5 { 1; } else { 0; };`
- ✅ `while` loop
- ✅ `for i := 0 to N { }` — for-to loop
- ✅ `for i in 0..N { }` — range loop
- ✅ `for item in array { }` — for-in array
- ✅ `for i, item in array { }` — indexed for-in
- ✅ `for key, value in map { }` — map iteration
- ✅ `break`
- ✅ `continue`

### Match
- ✅ Literal patterns
- ✅ Wildcard `_`
- ✅ Match on `Result`: `ok(v) => ..., err(e) => ...`
- ✅ Match on `Optional`: `some(v) => ..., none => ...`
- ✅ Match as expression
- ✅ Match on string values

### Functions
- ✅ Basic `fn name(args) -> RetType { }`
- ✅ Recursion
- ✅ Nested function definitions
- ✅ Return structs from functions
- ⚠️ Default params (compile but string defaults may not work on first call)
- ❌ Named arguments
- ❌ Void functions (no return type)

### Closures
- ✅ Anonymous fn: `f := fn(x: i64) -> i64 { x * 2 };`
- ✅ Capture variables
- ✅ Higher-order functions (pass/return closures)
- ✅ `make_adder` pattern
- ✅ `.map()` on arrays
- ✅ `.sort()` on arrays
- ❌ `.filter()` — parser issue with bool return type in closure

### Structs
- ✅ Definition with typed fields
- ✅ Construction `Struct { field: value }`
- ✅ Field access `s.field`
- ✅ Field mutation `s.field := value` and `s.field = value`
- ✅ Pass structs to functions
- ✅ Return structs from functions
- ⚠️ String fields work but f-string interpolation of string fields may show garbage

### Arrays
- ✅ Literal `[1, 2, 3]`
- ✅ Empty `[]`
- ✅ Fill syntax `[0; 10]`
- ✅ Indexing `arr[i]`
- ✅ `.push(val)`
- ✅ `.pop()`
- ✅ `.len`
- ✅ `.map()` with closure
- ✅ `.sort()`
- ❌ Slicing `arr[1..3]`
- ❌ `.join()`
- ❌ `.filter()` (bool return type issue)

### Maps
- ✅ Literal `{"key": value}`
- ✅ `.has("key")`
- ✅ `.len`
- ✅ `for key, value in map { }`
- ❌ `map["key"]` direct indexing (parser issue)
- ❌ `.keys()`
- ❌ `.values()`

### Result Type
- ✅ `ok(value)` constructor
- ✅ `err("message")` constructor
- ✅ `?` propagation operator
- ✅ `if r is ok(v) { }` pattern
- ✅ `if r is err(e) { }` pattern
- ✅ `match r { ok(v) => ..., err(e) => ... }`

### Optional Type
- ✅ `some(value)` constructor
- ✅ `none` value
- ✅ `?T` type annotation
- ✅ `if o is some(v) { }` pattern
- ✅ `if o is none { }` pattern
- ✅ `match o { some(v) => ..., none => ... }`

### Advanced Features
- ✅ `type Alias = i64;` — type aliases
- ✅ `async fn` / `await` — async functions
- ✅ `requires fs;` / `requires env;` — capabilities
- ✅ `soa Struct[N]` — struct of arrays
- ✅ Tensors — `tensor<f32>([shape], [data])`, `matmul()`, `.sum()`
- ❌ `try/catch` — not implemented

### Built-in Functions
- ✅ `println(string)` — print with newline
- ✅ `print(i64)` — print integer
- ✅ `print_string(s)` — print string without newline
- ✅ `print_f64(f)` — print float
- ✅ `str(i64)` — convert int to string
- ✅ `sqrt(f64)`, `sin(f64)`, `cos(f64)` — math builtins
- ✅ `pow(f64, f64)` — power function
- ✅ `PI`, `E` — math constants
- ✅ `f"interpolation {var}"` — f-strings

### String Methods
- ✅ `.len` — length
- ✅ `.upper()` — uppercase
- ✅ `.lower()` — lowercase
- ✅ `.trim()` — strip whitespace
- ✅ `.replace(old, new)` — replacement
- ✅ `.split(sep)` — split into array
- ✅ `.contains(sub)` — substring check
- ✅ `.starts_with(prefix)` — prefix check
- ✅ `.ends_with(suffix)` — suffix check
- ❌ `s[0..5]` — slicing syntax

## Code Generation Guidelines

For the EGDC code generator, use ONLY these patterns:

1. **Always** use `fn main() -> i64 { ... return 0; }` (never void main)
2. **Always** use `i64` for integers (avoid u32, u64)
3. **Never** use `bool` type annotation — use `:= true` or use `i64` (0/1)
4. **Never** use `!` operator — use `== 0` or `== false`
5. **Never** use `**` — use `pow(a, b)` for power
6. **Never** use try/catch — use `if r is err(e)` or match
7. **Never** use void functions — always `-> i64` with `return 0`
8. **Never** use named arguments — positional only
9. **Never** use `.filter()` on arrays — use manual loop + push pattern
10. **Never** use array/string slicing `[a..b]` — use loops
11. **Never** use `map["key"]` — use `for k,v in map` or `.has()`
12. **Never** use `.keys()` or `.values()` on maps — iterate with `for k,v`
13. **Never** use `.join()` on arrays — build string manually
14. **Never** use `as` for f64->i64 cast — keep types consistent
15. **Use** `for i in 0..N` for range loops (works!)
16. **Use** `for i := 0 to N` for counted loops (works!)
17. **Use** walrus `:=` for both binding and reassignment (works for both)
