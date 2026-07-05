use super::*;
use crate::synthesis;

/// Returns true when at least one example input is a string, structured type,
/// or pair. Scalar-only stages (enumerative expression search, scalar
/// expr_only gradient) cannot express those input shapes and would each burn
/// their full time budget before missing — the pipeline should skip straight
/// to the search-teacher / array path in that case.
fn has_non_scalar_input(problem: &Problem) -> bool {
    problem.examples.iter().any(|ex| {
        ex.inputs
            .iter()
            .any(|v| !matches!(v, Value::Int(_) | Value::Array(_)))
    })
}

/// Emit a learned string->string lookup table: `if s == "k" { return "v"; } ...
/// return "<default>";`. This is how an *arbitrary* string lexicon (e.g.
/// irregular inflection: have->has, be->is) is recovered — facts with no
/// transduction rule must be stored as a verified program.
fn code_string_string_map(fn_name: &str, default: &str, branches: &[(String, String)]) -> String {
    let q = |s: &str| format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""));
    let mut body = String::new();
    for (k, v) in branches {
        body.push_str(&format!(
            "    if s == {} {{\n        return {};\n    }}\n",
            q(k),
            q(v)
        ));
    }
    format!(
        "fn {fn_name}(s: string) -> string {{\n{body}    return {};\n}}\n",
        q(default)
    )
}

/// Build a whole-word string->string lookup-table program from single-arg
/// examples, or None if the mapping is not an arbitrary lexicon (multi-arg,
/// too few examples, inconsistent, or fewer than two distinct outputs). The
/// emitted program returns each trained output for its trained input by
/// construction, so it is correct on the training set without verification.
/// Shared by `solve_string_lexicon` (lib path) and the `--problem-json` CLI.
pub(super) fn string_lexicon_map_code(
    train: &[(Vec<String>, String)],
    fn_name: &str,
) -> Option<String> {
    use std::collections::BTreeMap;
    if train.len() < 3 || !train.iter().all(|(i, _)| i.len() == 1) {
        return None;
    }
    // Consistent whole-word map (same input -> one output).
    let mut map: BTreeMap<String, String> = BTreeMap::new();
    for (i, o) in train {
        match map.get(&i[0]) {
            Some(prev) if prev != o => return None,
            _ => {
                map.insert(i[0].clone(), o.clone());
            }
        }
    }
    // Need at least two distinct outputs (else a constant function).
    let distinct: std::collections::BTreeSet<&String> = map.values().collect();
    if distinct.len() < 2 {
        return None;
    }
    // Default = most frequent output; only off-default words get a branch.
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for (_, o) in train {
        *counts.entry(o.clone()).or_insert(0) += 1;
    }
    let default = counts
        .into_iter()
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
        .map(|(o, _)| o)?;
    let mut branches: Vec<(String, String)> =
        map.into_iter().filter(|(_, o)| *o != default).collect();
    branches.sort();
    Some(code_string_string_map(fn_name, &default, &branches))
}

/// Whole-word string->string lexicon teacher. Runs AFTER the suffix-transduction
/// specialist, so it only claims a problem when the mapping is an arbitrary
/// lookup no rule explains. The string sibling of `search_string_equality_map`.
fn solve_string_lexicon(
    problem: &Problem,
    train: &[(Vec<String>, String)],
    fn_name: &str,
) -> Option<SolveResult> {
    let code = string_lexicon_map_code(train, fn_name)?;
    crate::runtime::verify_problem_code(problem, &code).ok()?;
    Some(SolveResult {
        success: true,
        code,
        method: "string_lexicon_map".to_string(),
        error: None,
        metadata: Default::default(),
    })
}

/// Route string-output problems (signature `-> string`) to the string-program
/// path: the fast morphology specialist, then the general enumerative string
/// synthesizer. Returns None for non-string problems so the numeric pipeline runs.
/// FIELD-WISE STRUCT SYNTHESIS — the first path that PRODUCES a `Value::Struct`
/// output (previously the type was representable, renderable, and verifiable,
/// but nothing emitted it; struct_of_state benchmarks faked it with `Quad` +
/// hand-named structs).
///
/// Decomposition: a struct output is a TUPLE of independent functions of the
/// same inputs. For each field we project the examples onto that field and hand
/// the sub-problem to the FULL existing pipeline (`solve_problem` — scalar,
/// array, string machinery all apply), so every helper is independently
/// synthesized + verified. The assembled program is
///
///   struct S { f1: T1, ... }
///   fn name_f1(params) -> T1 { <verified sub-solve> }
///   fn name(params) -> S { return S { f1: name_f1(params), ... }; }
///
/// and the WHOLE assembly is strict-verified (interpreter executes the struct
/// constructor; `output_matches` compares runtime Struct vs wire Struct) before
/// being accepted — no fabrication on any failure. v1 scope: flat structs whose
/// fields are Int/Bool/Str/Array (nested structs decline honestly).
fn solve_struct_output(problem: &Problem) -> Option<SolveResult> {
    use crate::benchmark::Example;
    // Gate: every example's expected is a Struct with identical field names.
    let field_names: Vec<String> = match &problem.examples.first()?.expected {
        Value::Struct(fs) if !fs.is_empty() => fs.iter().map(|(n, _)| n.clone()).collect(),
        _ => return None,
    };
    for e in &problem.examples {
        match &e.expected {
            Value::Struct(fs)
                if fs.len() == field_names.len()
                    && fs.iter().zip(&field_names).all(|((n, _), fname)| n == fname) => {}
            _ => return None,
        }
    }
    // FUNCTIONAL-CONSISTENCY pre-gate: identical inputs must map to identical
    // outputs — a spec violating this is not a function and no search can save
    // it, so decline instantly instead of exhausting every stage per field.
    for (i, a) in problem.examples.iter().enumerate() {
        for b in problem.examples.iter().skip(i + 1) {
            if a.inputs == b.inputs && a.expected != b.expected {
                return None;
            }
        }
    }
    let field_type = |v: &Value| -> Option<&'static str> {
        match v {
            Value::Int(_) => Some("i64"),
            Value::Bool(_) => Some("bool"),
            Value::Str(_) => Some("string"),
            Value::Array(_) => Some("[i64]"),
            _ => None, // nested structs / exotic carriers: decline in v1
        }
    };

    let fn_name = problem.function_name();
    // Struct name from the signature's return type; a lowercase/absent one gets
    // a capitalized default derived from the fn name.
    let struct_name = problem
        .signature
        .rsplit("->")
        .next()
        .map(|s| s.trim().trim_end_matches('{').trim().to_string())
        .filter(|s| !s.is_empty() && s.chars().next().is_some_and(|c| c.is_ascii_uppercase()))
        .unwrap_or_else(|| {
            let mut n = fn_name.to_string();
            if let Some(c) = n.get_mut(0..1) {
                c.make_ascii_uppercase();
            }
            format!("{n}Out")
        });
    // Params: keep the repo signature's parameter list verbatim.
    let params_src = problem
        .signature
        .split_once('(')
        .and_then(|(_, r)| r.split_once(')'))
        .map(|(p, _)| p.trim().to_string())
        .unwrap_or_default();
    let param_names: Vec<String> = params_src
        .split(',')
        .filter_map(|p| p.split(':').next().map(|n| n.trim().to_string()))
        .filter(|n| !n.is_empty())
        .collect();
    if param_names.is_empty() {
        return None;
    }

    // One verified helper per field, via the FULL pipeline.
    let mut helpers = String::new();
    let mut decl_fields = Vec::new();
    let mut ctor_fields = Vec::new();
    let mut methods = Vec::new();
    for (idx, fname) in field_names.iter().enumerate() {
        let fvals: Vec<Value> = problem
            .examples
            .iter()
            .map(|e| match &e.expected {
                Value::Struct(fs) => fs[idx].1.clone(),
                _ => unreachable!("gated above"),
            })
            .collect();
        let fty = field_type(&fvals[0])?;
        let helper_name: &'static str =
            Box::leak(format!("{fn_name}_{fname}").into_boxed_str());
        let helper_sig: &'static str =
            Box::leak(format!("fn {helper_name}({params_src}) -> {fty}").into_boxed_str());
        let sub = Problem {
            name: helper_name.to_string(),
            category: "struct-field",
            description: "field-wise struct decomposition",
            signature: helper_sig,
            examples: problem
                .examples
                .iter()
                .zip(fvals.iter())
                .map(|(e, fv)| Example {
                    inputs: e.inputs.clone(),
                    expected: fv.clone(),
                })
                .collect(),
            ..Default::default()
        };
        let r = solve_problem(&sub);
        if !r.success {
            return None; // a field the engine can't synthesize ⇒ decline whole
        }
        helpers.push_str(r.code.trim_end());
        helpers.push_str("\n\n");
        decl_fields.push(format!("    {fname}: {fty},"));
        ctor_fields.push(format!("{fname}: {helper_name}({})", param_names.join(", ")));
        methods.push(r.method);
    }

    let code = format!(
        "struct {struct_name} {{\n{}\n}}\n\n{helpers}fn {fn_name}({params_src}) -> {struct_name} {{\n    return {struct_name} {{ {} }};\n}}\n",
        decl_fields.join("\n"),
        ctor_fields.join(", "),
    );
    // The WHOLE assembly must strict-verify — helpers were verified per-field,
    // but the constructor wiring is proven here, not assumed.
    if crate::runtime::verify_problem_code_strict(problem, &code).is_err() {
        return None;
    }
    Some(SolveResult {
        success: true,
        code,
        method: format!("struct_fieldwise({})", methods.join("+")),
        error: None,
        metadata: Default::default(),
    })
}

/// STRUCT-INPUT synthesis: FLATTEN-AND-WRAP. v1 scope: exactly ONE input, a
/// flat struct of Int fields, non-struct output. The fields become the params
/// of a flat sub-problem handed to the FULL pipeline; the assembly
///
///   struct Rect { h: i64, w: i64 }
///   fn area_core(h: i64, w: i64) -> T { <verified flat solve> }
///   fn area(r: Rect) -> T { return area_core(r.h, r.w); }
///
/// is strict-verified WHOLE (the wrapper harness renders struct arguments as
/// typed literals; the interpreter constructs + field-accesses them). Declines
/// honestly on any failure — never fabricates.
fn solve_struct_input(problem: &Problem) -> Option<SolveResult> {
    use crate::benchmark::Example;
    // Gate: one struct input with identical Int field names across examples.
    let field_names: Vec<String> = match problem.examples.first()?.inputs.as_slice() {
        [Value::Struct(fs)] if !fs.is_empty() => fs.iter().map(|(n, _)| n.clone()).collect(),
        _ => return None,
    };
    for e in &problem.examples {
        match e.inputs.as_slice() {
            [Value::Struct(fs)]
                if fs.len() == field_names.len()
                    && fs.iter().zip(&field_names).all(|((n, _), f)| n == f)
                    && fs.iter().all(|(_, v)| matches!(v, Value::Int(_))) => {}
            _ => return None,
        }
        if matches!(e.expected, Value::Struct(_)) {
            return None; // struct->struct: out of v1 scope
        }
    }

    let fn_name = problem.function_name();
    // Struct name from the signature's first param type, else a derived default.
    let struct_name = problem
        .signature
        .split_once('(')
        .and_then(|(_, r)| r.split_once(')'))
        .and_then(|(params, _)| params.split(',').next()?.split_once(':'))
        .map(|(_, ty)| ty.trim().to_string())
        .filter(|ty| ty.chars().next().is_some_and(|c| c.is_ascii_uppercase()))
        .unwrap_or_else(|| {
            let mut n = fn_name.to_string();
            if let Some(c) = n.get_mut(0..1) {
                c.make_ascii_uppercase();
            }
            format!("{n}In")
        });
    let ret_ty = problem
        .signature
        .rsplit("->")
        .next()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "i64".to_string());

    // Flat sub-problem: the fields ARE the params.
    let core_name: &'static str = Box::leak(format!("{fn_name}_core").into_boxed_str());
    let params_src = field_names
        .iter()
        .map(|f| format!("{f}: i64"))
        .collect::<Vec<_>>()
        .join(", ");
    let core_sig: &'static str =
        Box::leak(format!("fn {core_name}({params_src}) -> {ret_ty}").into_boxed_str());
    let sub = Problem {
        name: core_name.to_string(),
        category: "struct-input",
        description: "flattened struct-input core",
        signature: core_sig,
        examples: problem
            .examples
            .iter()
            .map(|e| Example {
                inputs: match e.inputs.as_slice() {
                    [Value::Struct(fs)] => fs.iter().map(|(_, v)| v.clone()).collect(),
                    _ => unreachable!("gated above"),
                },
                expected: e.expected.clone(),
            })
            .collect(),
        ..Default::default()
    };
    let r = solve_problem(&sub);
    if !r.success {
        return None;
    }

    let decl_fields = field_names
        .iter()
        .map(|f| format!("    {f}: i64,"))
        .collect::<Vec<_>>()
        .join("\n");
    let access = field_names
        .iter()
        .map(|f| format!("r.{f}"))
        .collect::<Vec<_>>()
        .join(", ");
    let code = format!(
        "struct {struct_name} {{\n{decl_fields}\n}}\n\n{}\n\nfn {fn_name}(r: {struct_name}) -> {ret_ty} {{\n    return {core_name}({access});\n}}\n",
        r.code.trim_end(),
    );
    if crate::runtime::verify_problem_code_strict(problem, &code).is_err() {
        return None;
    }
    Some(SolveResult {
        success: true,
        code,
        method: format!("struct_input_flatten({})", r.method),
        error: None,
        metadata: Default::default(),
    })
}

fn solve_string_output(problem: &Problem) -> Option<SolveResult> {
    if !problem
        .signature
        .replace(' ', "")
        .to_ascii_lowercase()
        .contains("->string")
    {
        return None;
    }
    // Examples must be all-string-input, string-output.
    let to_rows = |exs: &[crate::benchmark::Example]| -> Option<Vec<(Vec<String>, String)>> {
        exs.iter()
            .map(|e| {
                let ins: Option<Vec<String>> = e
                    .inputs
                    .iter()
                    .map(|v| match v {
                        Value::Str(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect();
                match (&ins, &e.expected) {
                    (Some(i), Value::Str(o)) => Some((i.clone(), o.clone())),
                    _ => None,
                }
            })
            .collect()
    };
    let train = to_rows(&problem.examples)?;
    // Holdouts are evaluator-owned and must not influence rule selection.
    let holds = Vec::new();
    let fn_name = problem.function_name();
    let single = train.iter().all(|(i, _)| i.len() == 1);

    // Fast morphology specialist (single-arg suffix transduction). A genuine
    // generalizing transducer, so it stays first.
    if single {
        use crate::morph_transduce::{solve_morph_transduction, StrExample};
        let mk = |rs: &[(Vec<String>, String)]| {
            rs.iter()
                .map(|(i, o)| StrExample {
                    input: i[0].clone(),
                    expected: o.clone(),
                })
                .collect::<Vec<_>>()
        };
        let m = solve_morph_transduction(fn_name, &mk(&train), &mk(&holds));
        if m.success {
            return Some(SolveResult {
                success: true,
                code: m.code,
                method: m.method,
                error: None,
                metadata: Default::default(),
            });
        }
    }

    // General enumerative string synthesizer. NL-BRIDGE-1: this runs BEFORE the
    // whole-word lexicon fallback. `string_synth` finds a GENERALIZING rule
    // (`s.to_uppercase()`, `reverse`, slice, affix, …) that holds on unseen inputs;
    // the lexicon below only MEMORIZES the training pairs in an if-chain. Trying the
    // generalizer first means a request like "uppercase a string" emits the real
    // `s.to_uppercase()` program instead of a lookup table that fails on a fresh
    // string — memorization is the irregular-map fallback, not the default.
    use crate::string_synth::{synthesize_string_program, StrSynthExample};
    let params: Vec<String> = problem
        .signature
        .split_once('(')
        .and_then(|(_, r)| r.split_once(')'))
        .map(|(p, _)| p)
        .unwrap_or("")
        .split(',')
        .filter_map(|p| p.split(':').next().map(|n| n.trim().to_string()))
        .filter(|n| !n.is_empty())
        .collect();
    let params = if params.is_empty() {
        vec!["s".to_string()]
    } else {
        params
    };
    let all: Vec<StrSynthExample> = train
        .iter()
        .chain(holds.iter())
        .map(|(i, o)| StrSynthExample {
            inputs: i.clone(),
            expected: o.clone(),
        })
        .collect();
    let r = synthesize_string_program(&params, &all);
    if r.success {
        let code = r
            .code
            .replacen("fn transform(", &format!("fn {fn_name}("), 1);
        return Some(SolveResult {
            success: true,
            code,
            method: r.method,
            error: None,
            metadata: Default::default(),
        });
    }

    // Whole-word lexicon lookup (LAST resort): for arbitrary string->string maps no
    // suffix transduction OR generalizing string program explains (e.g. irregular
    // inflection: have->has, be->is). This memorizes the training pairs, so it is
    // tried only after the generalizers above decline.
    if single {
        if let Some(result) = solve_string_lexicon(problem, &train, fn_name) {
            return Some(result);
        }
    }
    None
}

pub(super) fn solve_problem(problem: &Problem) -> SolveResult {
    // Suppress cache recording while the analogy universal re-fitter is
    // re-solving a TEACHER-AUGMENTED problem: that problem's examples are
    // synthetic (perturbed teacher samples) with a fingerprint no real query
    // will ever match, so recording it only pollutes the donor pool. Only the
    // verified-against-ORIGINAL emit from `analogy_solve` should be cached, and
    // it is (under the real query's fingerprint, outside this guard).
    let recordable = !super::analogy::in_refit() && !crate::learning_freeze::is_frozen();

    // REGISTRY PRIMITIVES prefer the domain-direct string emitter first:
    // string_synth / morph emit `s.trim()`-style TRANSPILABLE bodies, while the
    // decompose lane below can win the same spec with an interpreter-only char
    // loop that breaks downstream Rust transpilation (observed post-merge:
    // trim -> decompose-char-filter, `for ch in s` / `""` / char-vs-&str compile
    // errors in every component build). The bench cascade for every other
    // category is untouched.
    if problem.category == "registry-op" {
        if let Some(result) = solve_string_output(problem) {
            if result.success && recordable {
                crate::solved_cache::record(problem, &result.method, &result.code);
            }
            return result;
        }
    }

    // Structural decomposition at the VERY TOP: ms-scale, self-gating, and both
    // the string-output path (string_synth enumerative burn) and the int-array
    // frontier below it consume the whole per-task budget before a cheap
    // decompose shape (char-filter, derivative, rolling_max) is ever attempted —
    // the same starvation disease as the library/float lanes, same cure.
    if let Some(result) = super::search_decompose::try_decompose(problem) {
        if recordable {
            crate::solved_cache::record(problem, &result.method, &result.code);
            crate::op_library::maybe_record_learned(problem, &result);
        }
        return result;
    }

    // String-output problems take the additive string-program path (the i64
    // gradient/search pipeline cannot express string outputs).
    if let Some(result) = solve_string_output(problem) {
        if result.success && recordable {
            crate::solved_cache::record(problem, &result.method, &result.code);
            crate::op_library::maybe_record_learned(problem, &result);
        }
        return result;
    }

    // Verified reference-op library FIRST — a known algorithm that reproduces
    // every example returns instantly (behaviour-matched, cheap). Runs here,
    // AHEAD of the array-frontier search below, because many array->array tasks
    // (move-zeros, swap-adjacent, consecutive-sums, rotate, …) have an exact
    // library op but were timing out inside `synthesize_array`'s frontier before
    // `try_library` (which lived only in the scalar `solve_problem_inner`) could
    // run. Non-array problems still hit the scalar-path `try_library` too — a
    // second call is a few ms and never changes the result.
    if let Some(result) = crate::op_library::try_library(problem) {
        if recordable {
            crate::solved_cache::record(problem, &result.method, &result.code);
        }
        return result;
    }

    // Fitness-guided evolutionary synthesis (no model; gated NSYNTH_EVOLVE): hill-
    // climbs an accumulator-loop program for fold-shaped tasks the library/search
    // miss. Verifier-gated; feeds the flywheel via the success hooks below.
    if let Some(result) = crate::synth_evolve::synthesize_evolve(problem) {
        if recordable {
            crate::solved_cache::record(problem, &result.method, &result.code);
            crate::op_library::maybe_record_learned(problem, &result);
        }
        return result;
    }

    // Struct-output problems: FIELD-WISE DECOMPOSITION. Each field becomes an
    // independent sub-problem solved by this whole pipeline; the assembled
    // program (struct decl + verified per-field helpers + a constructor fn) is
    // strict-verified as a whole before being accepted.
    if let Some(result) = solve_struct_output(problem) {
        if result.success && recordable {
            crate::solved_cache::record(problem, &result.method, &result.code);
        }
        return result;
    }

    // Struct-INPUT problems: FLATTEN-AND-WRAP. The struct's fields become the
    // params of a flat sub-problem solved by this whole pipeline; a wrapper fn
    // destructures the struct (r.w, r.h) into the verified core, and the whole
    // (decl + core + wrapper) is strict-verified before acceptance.
    if let Some(result) = solve_struct_input(problem) {
        if result.success && recordable {
            crate::solved_cache::record(problem, &result.method, &result.code);
        }
        return result;
    }

    // Array-output problems (`[i64] -> [i64]`): exact array_transform before scalar paths.
    let array_io = problem.examples.first().is_some_and(|ex| {
        ex.inputs.len() == 1
            && matches!(ex.inputs[0], Value::Array(_))
            && matches!(ex.expected, Value::Array(_))
    });
    if array_io {
        let t_arr = std::time::Instant::now();
        if let Some(result) = synthesis::synthesize_array(problem) {
            if result.success {
                eprintln!(
                    "[solve] early array_output OK in {:.3}s — {}",
                    t_arr.elapsed().as_secs_f32(),
                    result.method
                );
                if recordable {
                    crate::solved_cache::record(problem, &result.method, &result.code);
                    crate::op_library::maybe_record_learned(problem, &result);
                }
                return result;
            }
        }
        eprintln!(
            "[solve] early array_output MISS in {:.3}s",
            t_arr.elapsed().as_secs_f32()
        );
    }

    let result = solve_problem_inner(problem);
    if result.success && recordable {
        // Record every successful solve. De-duped inside the cache so reruns
        // don't re-write the same entry. Persisted via `solved_cache::flush`
        // which callers (bench runner, main) invoke at shutdown.
        crate::solved_cache::record(problem, &result.method, &result.code);
        // Loop 2 flywheel: a novel verified program becomes a runtime library op
        // (gated on NSYNTH_LEARNED_OPS_PATH; inert otherwise).
        crate::op_library::maybe_record_learned(problem, &result);
    }
    result
}

/// Optional global per-solve budget (ms). When set, stages are skipped once it is
/// exhausted so a solve degrades gracefully instead of spinning (profiling showed
/// doomed searches burning 15-30s without solving). Also caps the teacher stage
/// (see `strategy::teacher_budget_sec`). Unset -> unbounded (legacy behavior).
fn solve_budget_ms() -> Option<u128> {
    std::env::var("NSYNTH_SOLVE_BUDGET_MS")
        .ok()
        .and_then(|s| s.parse().ok())
}

fn budget_miss() -> SolveResult {
    SolveResult {
        success: false,
        code: String::new(),
        method: "budget-exhausted".to_string(),
        error: Some("NSYNTH_SOLVE_BUDGET_MS exhausted".to_string()),
        metadata: DifferentiableMetadata::default(),
    }
}

fn solve_problem_inner(problem: &Problem) -> SolveResult {
    let t0 = std::time::Instant::now();
    let over_budget = |t: &std::time::Instant| solve_budget_ms().is_some_and(|b| t.elapsed().as_millis() > b);
    // Under a global budget, cap ALL gradient training (the array/native cores are
    // the dominant time-sink — profiling showed synthesize_array burning ~11s on a
    // doomed task) via the thread-local train deadline. RAII: held for this solve.
    let _train_deadline = solve_budget_ms()
        .map(|ms| crate::synthesis::common::TrainDeadline::set(std::time::Duration::from_millis(ms as u64)));

    // Float (continuous) lane first: a `-> f64` problem is least-squares affine
    // regression, a different regime from the exact-integer machinery below
    // (which would choke on f64 inputs). Self-gates to f64 signatures and returns
    // None for everything else, so integer problems are untouched.
    if let Some(result) =
        super::search_float::search_float_affine(problem, &problem.function_name())
    {
        if result.success {
            return result;
        }
    }
    // Polynomial float lane: affine refuses `πr²` / `(4/3)πr³` / `k·a·b` — the
    // geometry formulas MBPP's float tasks are made of. Parsimony-laddered power
    // products, over-determination-gated, same round-then-reverify contract.
    if let Some(result) = super::search_float::search_float_poly(problem, &problem.function_name())
    {
        if result.success {
            return result;
        }
    }

    let router_ctx = post_enumerative_context(problem);

    // Stage 0: persistent cross-run memoization. A previous solve with the
    // same I/O fingerprint (+ a live re-verification to guard against
    // fingerprint collisions / stale cache) returns instantly. This is the
    // first place cross-run "learning" shows up — every successful solve
    // below is recorded and will hit on the next run.
    let mut cached_fallback: Option<crate::solved_cache::CachedSolution> = None;
    if let Some(cached) = crate::solved_cache::lookup(problem) {
        if should_bypass_solved_cache(problem, &router_ctx, &cached) {
            eprintln!(
                "[solve] bypassing solved_cache HIT in {:.3}s — {}",
                t0.elapsed().as_secs_f32(),
                cached.method
            );
            cached_fallback = Some(cached);
        } else {
            eprintln!(
                "[solve] solved_cache HIT in {:.3}s — {}",
                t0.elapsed().as_secs_f32(),
                cached.method
            );
            return SolveResult {
                success: true,
                code: cached.code,
                method: cached.method,
                error: None,
                metadata: DifferentiableMetadata::default(),
            };
        }
    }

    // Verified reference-op library FIRST (after the exact cache): a KNOWN algorithm
    // (is_prime / gcd / factorial / power / count_value / …) whose impl reproduces
    // EVERY example. Runs before the search/affine/gradient stages because those can
    // OVERFIT a tiny example set (2 seed points fit spuriously by a single-branch
    // affine) and short-circuit the correct algorithm. Behavior-matched + example-
    // verified, so it fires ONLY on a genuine match; a spurious partial match is
    // caught by the caller's holdout re-verification.
    if let Some(result) = crate::op_library::try_library(problem) {
        eprintln!("[solve] library OK in {:.3}s — {}", t0.elapsed().as_secs_f32(), result.method);
        return result;
    }

    // Composition tier: a CHAIN of 2-3 verified library ops (typed, value-level,
    // OE-pruned — see `op_pipeline`). Runs right after the single-op library and
    // before the search/affine/gradient stages for the same reason: a genuine
    // known-algorithm composition ("sum of digits of n!") must not be starved or
    // overfit-shadowed by bounded expression search. Never fires for a single-op
    // fit, so library/search attribution is unaffected.
    if let Some(result) = crate::op_pipeline::try_pipeline(problem) {
        eprintln!(
            "[solve] library-pipeline OK in {:.3}s — {}",
            t0.elapsed().as_secs_f32(),
            result.method
        );
        return result;
    }

    // Fixed-K TUPLE output tier: `(min, max)`-style tasks whose every output is a
    // constant-length scalar array. Solves each output column by reusing the
    // library/pipeline tiers above, then assembles a verified multi-fn program.
    // Self-gates (instant None) for any non-fixed-array output, so scalar/array
    // problems are untouched. Runs here so a genuine tuple task is not first
    // mangled by the variable-length array machinery below.
    if let Some(result) = super::search_tuple::try_tuple(problem) {
        eprintln!("[solve] tuple-columns OK in {:.3}s — {}", t0.elapsed().as_secs_f32(), result.method);
        return result;
    }

    // Structural DECOMPOSITION tier (emergent-ops step 1): map/filter/select
    // hypotheses over list I/O, element holes solved by reusing the verified op
    // library as a COMPONENT basis + data-mined predicates. Self-gates to
    // single-list-input problems; end-to-end re-verified inside. Runs early for
    // the same reason as the tuple tier — its shapes must not be starved by the
    // int-array machinery below (string-element lists especially).
    if let Some(result) = super::search_decompose::try_decompose(problem) {
        eprintln!("[solve] decompose OK in {:.3}s — {}", t0.elapsed().as_secs_f32(), result.method);
        return result;
    }

    // Exact multi-argument linear family first: a 2-3 arg affine or
    // single-threshold-affine rule is solved by a direct integer linear solve in
    // microseconds and verified against every example, so it must short-circuit
    // ahead of the search/gradient stages rather than risk being starved by an
    // earlier candidate. No-op (instant None) for 1-arg or non-linear data.
    if let Some(result) = super::search::solve_multi_arg_affine(problem) {
        if result.success {
            eprintln!(
                "[solve] multi-arg affine OK in {:.3}s — {}",
                t0.elapsed().as_secs_f32(),
                result.method
            );
            return result;
        }
    }

    let non_scalar = has_non_scalar_input(problem);

    // Run the cheap preemptive search teacher first — it's ms-scale and
    // covers a large fraction of the benchmark. This is the correct pre-
    // gradient stage ordering; running CachedTeachers before this was a
    // measured regression (curve_analysis flagged 22-24s cumulative wall-
    // clock on problems the preemptive stage solved in 5-10ms).
    let preemptive_search_result = solve_problem_from_preemptive_search_teacher(problem);

    // Stage 1.5: cross-run knowledge transfer via CachedTeachers. Runs only
    // when cheaper stages (Stage 0 exact-match + preemptive search) have
    // already missed. Bounded by NSYNTH_TEACHER_BUDGET_SEC (default 15s).
    //
    // This is the "gradient distillation from prior solves" path — real work,
    // not a lookup. Placing it after preemptive search means fast problems
    // stay fast; placing it before enumerative + synth_gradient means a
    // transfer win short-circuits the slowest stages of the pipeline.
    if !non_scalar && preemptive_search_result.is_none() && crate::solved_cache::entry_count() > 0 {
        let strategy = crate::strategy::CachedTeachers;
        if <crate::strategy::CachedTeachers as crate::strategy::SynthesisStrategy>::applicable(
            &strategy, problem,
        ) {
            let t_teacher = std::time::Instant::now();
            if let Some(result) =
                <crate::strategy::CachedTeachers as crate::strategy::SynthesisStrategy>::try_solve(
                    &strategy, problem,
                )
            {
                if result.success {
                    eprintln!(
                        "[solve] cached_teachers OK in {:.1}s",
                        t_teacher.elapsed().as_secs_f32()
                    );
                    return result;
                }
            }
            eprintln!(
                "[solve] cached_teachers MISS in {:.1}s",
                t_teacher.elapsed().as_secs_f32()
            );
        }
    }

    // Stage 1.6: analogy-driven transfer (Phase 3.2), opt-in via NSYNTH_ANALOGY=1.
    // Runs alongside CachedTeachers (does not replace it) so it can be A/B'd
    // without regressing the proven path. Universal: transfers donors of ANY
    // type (scalar/array/tree/string/…) via rename+verify, plus type-specific
    // re-fit fallback. Not gated on scalar-only. Emits only verifier-accepted
    // code.
    if std::env::var("NSYNTH_ANALOGY").as_deref() == Ok("1")
        && !super::analogy::in_refit()
        && preemptive_search_result.is_none()
        && crate::solved_cache::entry_count() > 0
    {
        let t_analogy = std::time::Instant::now();
        if let Some(result) = super::analogy::analogy_solve(problem) {
            if result.success {
                eprintln!(
                    "[solve] analogy OK ({}) in {:.1}s",
                    result.method,
                    t_analogy.elapsed().as_secs_f32()
                );
                return result;
            }
        }
        eprintln!(
            "[solve] analogy MISS in {:.1}s",
            t_analogy.elapsed().as_secs_f32()
        );
    }

    // Probabilistic synthesis: detect and synthesize probabilistic programs
    // using MCMC inference for uncertainty/randomness. Runs after CachedTeachers
    // (which are fast) but before enumerative (which is expensive). Only activates
    // when examples suggest non-deterministic behavior (conflicting outputs,
    // sampling patterns, uncertainty).
    if super::probabilistic::is_probabilistic_problem(problem) {
        eprintln!("[solve] trying probabilistic synthesis");
        let prob_result = super::probabilistic::solve_probabilistic_problem(problem);
        if prob_result.success {
            eprintln!("[solve] probabilistic OK");
            return prob_result;
        }
        eprintln!("[solve] probabilistic failed, continuing to normal pipeline");
    }

    // Stage 1: Classical bottom-up symbolic enumeration (no neural weights, no
    // gradients). Cheap on small closed-form expressions and cuts them off
    // before the slower gradient path even starts. The actual "brain" —
    // neural-style continuous parameter search — lives in the differentiable
    // synth_gradient / univ_arr_gradient stages further down.
    //
    // Skipped for inputs this enumerator cannot express (Str, Pair, struct):
    // those are handled by downstream search teachers and would otherwise
    // burn the full enumerative budget only to miss.
    if over_budget(&t0) {
        eprintln!("[solve] budget exhausted before enumerative in {:.1}s", t0.elapsed().as_secs_f32());
        return budget_miss();
    }
    if should_try_enumerative(
        problem,
        &router_ctx,
        non_scalar,
        preemptive_search_result.is_some(),
    ) {
        let t_enum = std::time::Instant::now();
        if let Some(result) = crate::enumerative::synthesize_enumerative(problem) {
            if result.success {
                eprintln!(
                    "[solve] enumerative OK in {:.1}s",
                    t_enum.elapsed().as_secs_f32()
                );
                method_router::record_win(problem, ROUTE_ENUMERATIVE);
                return result;
            }
        }
        // A MISS here means the enumerator exhausted THIS call's time budget,
        // not that the problem is impossible: its search frontier is persisted
        // (keyed by the examples fingerprint), so a later solve of the same
        // problem resumes deeper rather than restarting from size 1.
        eprintln!(
            "[solve] enumerative MISS (budget exhausted, frontier persisted) in {:.1}s",
            t_enum.elapsed().as_secs_f32()
        );
        method_router::record_miss(problem, ROUTE_ENUMERATIVE);
    } else if non_scalar {
        eprintln!("[solve] skipping enumerative: non-scalar input (string/pair/struct)");
    } else if preemptive_search_result.is_some() {
        eprintln!("[solve] skipping enumerative: exact search preemption");
    } else {
        let ranked = normalized_router_stats(problem, &router_ctx);
        if let Some(top) = ranked.first().copied() {
            eprintln!(
                "[solve] skipping enumerative: method_router favors {} ({} wins, {}% success)",
                top.route,
                top.wins,
                top.success_rate_percent()
            );
        } else {
            eprintln!("[solve] skipping enumerative: method_router preemption");
        }
    }

    if over_budget(&t0) {
        eprintln!("[solve] budget exhausted before post-enumeration in {:.1}s", t0.elapsed().as_secs_f32());
        if let Some(cached) = cached_fallback {
            return SolveResult {
                success: true,
                code: cached.code,
                method: cached.method,
                error: None,
                metadata: DifferentiableMetadata::default(),
            };
        }
        return budget_miss();
    }
    let result = solve_problem_after_enumeration(problem, t0, preemptive_search_result);
    if result.success {
        return result;
    }
    if let Some(cached) = cached_fallback {
        eprintln!("[solve] restoring solved_cache fallback after routed miss");
        return SolveResult {
            success: true,
            code: cached.code,
            method: cached.method,
            error: None,
            metadata: DifferentiableMetadata::default(),
        };
    }
    result
}

#[cfg(test)]
mod string_lexicon_tests {
    use crate::benchmark::{Example, Problem, Value};
    use crate::solver::solve_problem;

    fn str_str_problem(rows: &[(&str, &str)]) -> Problem {
        Problem {
            name: "irregular_3sg".to_string(),
            category: "comprehension",
            description: "",
            signature: "fn irregular_3sg(s: string) -> string",
            examples: rows
                .iter()
                .map(|(i, o)| Example {
                    inputs: vec![Value::Str((*i).to_string())],
                    expected: Value::Str((*o).to_string()),
                })
                .collect(),
            holdouts: vec![],
            reference_code: "",
            synthetic_args: Vec::new(),
            synthetic_values: Vec::new(),
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: vec![],
        }
    }

    // Irregular inflection is an arbitrary string->string lexicon no suffix rule
    // explains; the string-lexicon teacher must recover it as a verified lookup.
    #[test]
    fn string_lexicon_recovers_irregular_inflection() {
        let problem = str_str_problem(&[
            ("have", "has"),
            ("be", "is"),
            ("do", "does"),
            ("go", "goes"),
            ("walk", "-"),
            ("read", "-"),
            ("write", "-"),
            ("help", "-"),
            ("open", "-"),
            ("call", "-"),
        ]);
        let result = solve_problem(&problem);
        assert!(result.success, "failed to recover the irregular lexicon");
        assert_eq!(
            result.method, "string_lexicon_map",
            "expected the string-lexicon teacher, got {}",
            result.method
        );
        assert!(result.code.contains("if s == \"have\""));
        assert!(result.code.contains("return \"has\";"));
        // The regular majority is the default sentinel, not an explicit branch.
        assert!(!result.code.contains("if s == \"walk\""));
    }
}

#[cfg(test)]
mod struct_output_tests {
    use super::solve_struct_output;
    use crate::benchmark::{Example, Problem, Value};

    /// NO FABRICATION, instantly: a spec that is not a FUNCTION (identical
    /// input demanding different structs) is declined by the functional-
    /// consistency pre-gate — no search, no made-up constructor.
    #[test]
    fn struct_path_declines_non_functional_spec_instantly() {
        let mk = |x: i64, noise: i64| Example {
            inputs: vec![Value::Int(x)],
            expected: Value::Struct(vec![
                ("ok".to_string(), Value::Int(x + 1)),
                ("noise".to_string(), Value::Int(noise)),
            ]),
        };
        let problem = Problem {
            name: "chaos".into(),
            category: "struct-unit",
            description: "non-functional spec",
            signature: "fn chaos(x: i64) -> Chaos",
            // SAME input x=1 demands noise=17 AND noise=-940: not a function.
            examples: vec![mk(1, 17), mk(1, -940), mk(3, 5)],
            ..Default::default()
        };
        let started = std::time::Instant::now();
        assert!(solve_struct_output(&problem).is_none(), "must decline");
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "decline must be the pre-gate, not budget exhaustion"
        );
    }

    /// Mixed field types + inconsistent field NAMES across examples decline.
    #[test]
    fn struct_path_declines_inconsistent_field_names() {
        let problem = Problem {
            name: "shape".into(),
            category: "struct-unit",
            description: "inconsistent fields",
            signature: "fn shape(x: i64) -> Shape",
            examples: vec![
                Example {
                    inputs: vec![Value::Int(1)],
                    expected: Value::Struct(vec![("a".to_string(), Value::Int(2))]),
                },
                Example {
                    inputs: vec![Value::Int(2)],
                    expected: Value::Struct(vec![("b".to_string(), Value::Int(3))]),
                },
            ],
            ..Default::default()
        };
        assert!(solve_struct_output(&problem).is_none());
    }
}
