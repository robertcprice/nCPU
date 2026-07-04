//! TYPED BOTTOM-UP ENUMERATION — the universal expression search (power-arc
//! move 1).
//!
//! The remaining benchmark misses are small compositions over strings and
//! lists (`anti_shuffle` = join(map(split(x, " "), sort_chars), " ")) that no
//! behavior-matched op or hand schema reaches, because the program space was
//! never SEARCHABLE for those types. This module makes it searchable the
//! standard way (BUSTLE-without-the-model):
//!
//!   * a pool of typed expressions {Str, StrList, Int} grown by depth over a
//!     small Mog-emittable operator surface,
//!   * OBSERVATIONAL EQUIVALENCE dedup — two expressions with identical output
//!     vectors across all examples are the same search state,
//!   * MAP as a second-order operator drawing its element function from the
//!     SAME pool (a Str→Str expression over a fresh variable), which is what
//!     lets split→map(sort_chars)→join emerge at depth 4,
//!   * acceptance only when an expression's outputs equal EVERY example's
//!     expected value, then Mog emission + the caller's end-to-end re-verify.
//!
//! Soundness contract unchanged: recover-or-refuse, full-example verification,
//! bounded work (pool caps + depth ≤ 4).

use crate::benchmark::{Problem, Value};
use crate::solver::SolveResult;
use std::collections::HashSet;

/// A typed value in the enumeration domain.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum V {
    S(String),
    L(Vec<String>),
    I(i64),
}

/// An expression: how to compute it (interpreted on V) and how to emit it as a
/// Mog expression over the input variable `s` (plus helper fns for map bodies).
#[derive(Clone)]
struct Expr {
    /// Output on each example input (the OE signature).
    outs: Vec<V>,
    /// Mog source for this expression, with `s` as the input variable. Map
    /// expressions reference a helper `fn eXX(w: string) -> string` emitted
    /// alongside (collected via `helpers`).
    mog: String,
    /// Helper fn definitions this expression depends on (map bodies).
    helpers: Vec<String>,
    depth: usize,
}

const MAX_POOL_PER_TYPE: usize = 600;
const MAX_DEPTH: usize = 4;
/// DETERMINISTIC work budget: total expressions generated across all depths.
/// The narrow surface stayed small by luck; a wider one blew past the per-task
/// wall and STARVED downstream solvers (46->32 regression, reverted). A node
/// counter (not wall-clock — CPU-load-flaky) bounds the search so widening the
/// operator set can never again cost the whole time budget.
const MAX_NODES: usize = 6000;

/// Solve a single-STRING-input problem whose output is a string, by typed
/// enumeration. Returns a full Mog program (entry + helpers) on success.
pub(super) fn try_typed_enum_str(problem: &Problem, name: &str) -> Option<SolveResult> {
    let inputs: Vec<String> = problem
        .examples
        .iter()
        .map(|ex| {
            if let [Value::Str(s)] = ex.inputs.as_slice() {
                Some(s.clone())
            } else {
                None
            }
        })
        .collect::<Option<_>>()?;
    let expected: Vec<V> = problem
        .examples
        .iter()
        .map(|ex| match &ex.expected {
            Value::Str(s) => Some(V::S(s.clone())),
            Value::Int(i) => Some(V::I(*i)),
            _ => None,
        })
        .collect::<Option<_>>()?;

    // Depth-1 seeds: the input itself.
    let seed = Expr {
        outs: inputs.iter().map(|s| V::S(s.clone())).collect(),
        mog: "s".to_string(),
        helpers: vec![],
        depth: 1,
    };
    let mut pool: Vec<Expr> = vec![seed];
    let mut seen: HashSet<Vec<V>> = HashSet::new();
    seen.insert(pool[0].outs.clone());

    // Str→Str element transforms usable as map bodies (applied to a fresh
    // variable `w`). Each is (apply, mog-body). Kept small and total.
    type ElemOp = (&'static str, fn(&str) -> String);
    let elem_ops: [ElemOp; 5] = [
        ("w.upper()", |w| w.to_uppercase()),
        ("w.lower()", |w| w.to_lowercase()),
        ("w.reverse()", |w| w.chars().rev().collect()),
        ("sort_chars", |w| {
            let mut cs: Vec<char> = w.chars().collect();
            cs.sort_unstable();
            cs.into_iter().collect()
        }),
        ("identity", |w| w.to_string()),
    ];

    // HOLE POWER (emergent): additional (string)->string transforms drawn from
    // the EXISTING verified op library, applied by execution. No hand-list — the
    // library IS the surface, and it grows with the corpus. GUIDED best-first
    // pruning (below) keeps the combined 5-core + N-library set within budget.
    let lib_str_ops: Vec<(&'static str, &'static str)> = crate::op_library::OPS
        .iter()
        .filter(|op| op.arity == 1 && op_returns(op.mog, "string") && op_takes_string(op.mog))
        .map(|op| (op.name, op.mog))
        .collect();
    let apply_lib = |mog: &str, name: &str, s: &str| -> Option<String> {
        match crate::runtime::execute_function(mog, name, &[Value::Str(s.to_string())], name) {
            Ok(crate::runtime::Value::Str(out)) => Some(out),
            _ => None,
        }
    };
    // Expected strings for goal scoring (empty for non-Str outputs).
    let expected_str: Vec<String> = expected
        .iter()
        .map(|v| match v {
            V::S(s) => s.clone(),
            _ => String::new(),
        })
        .collect();

    let mut nodes = 0usize;
    for depth in 2..=MAX_DEPTH {
        if nodes >= MAX_NODES {
            break;
        }
        let prev: Vec<Expr> = pool.iter().filter(|e| e.depth == depth - 1).cloned().collect();
        let mut fresh: Vec<Expr> = Vec::new();

        for e in &prev {
            if nodes >= MAX_NODES {
                break;
            }
            nodes += 1;
            match &e.outs[0] {
                V::S(_) => {
                    // Unary string transforms.
                    for (body, f) in &elem_ops {
                        if *body == "identity" {
                            continue;
                        }
                        let outs: Vec<V> = e
                            .outs
                            .iter()
                            .map(|v| {
                                let V::S(s) = v else { unreachable!() };
                                V::S(f(s))
                            })
                            .collect();
                        let mog = match *body {
                            "sort_chars" => format!("sortchars({})", e.mog),
                            b => format!("{}", b.replace("w.", &format!("{}.", e.mog))),
                        };
                        let mut helpers = e.helpers.clone();
                        if *body == "sort_chars" {
                            helpers.push(SORTCHARS_FN.to_string());
                        }
                        fresh.push(Expr { outs, mog, helpers, depth });
                    }
                    // Library (string)->string ops (hole power), applied by exec.
                    for &(op_name, op_mog) in &lib_str_ops {
                        let mut outs = Vec::with_capacity(e.outs.len());
                        let mut ok = true;
                        for v in &e.outs {
                            let V::S(s) = v else { unreachable!() };
                            match apply_lib(op_mog, op_name, s) {
                                Some(o) => outs.push(V::S(o)),
                                None => {
                                    ok = false;
                                    break;
                                }
                            }
                        }
                        if !ok {
                            continue;
                        }
                        let mut helpers = e.helpers.clone();
                        helpers.push(op_mog.to_string());
                        fresh.push(Expr { outs, mog: format!("{op_name}({})", e.mog), helpers, depth });
                    }
                    // split on " " -> StrList.
                    let outs: Vec<V> = e
                        .outs
                        .iter()
                        .map(|v| {
                            let V::S(s) = v else { unreachable!() };
                            V::L(s.split(' ').map(str::to_string).collect())
                        })
                        .collect();
                    fresh.push(Expr {
                        outs,
                        mog: format!("{}.split(\" \")", e.mog),
                        helpers: e.helpers.clone(),
                        depth,
                    });
                    // length -> Int.
                    let outs: Vec<V> = e
                        .outs
                        .iter()
                        .map(|v| {
                            let V::S(s) = v else { unreachable!() };
                            V::I(s.chars().count() as i64)
                        })
                        .collect();
                    fresh.push(Expr {
                        outs,
                        mog: format!("{}.len", e.mog),
                        helpers: e.helpers.clone(),
                        depth,
                    });
                }
                V::L(_) => {
                    // map(elem_op) over the list — the second-order step.
                    for (body, f) in &elem_ops {
                        if *body == "identity" {
                            continue;
                        }
                        let outs: Vec<V> = e
                            .outs
                            .iter()
                            .map(|v| {
                                let V::L(l) = v else { unreachable!() };
                                V::L(l.iter().map(|w| f(w)).collect())
                            })
                            .collect();
                        let helper_name = format!("em{}", hash8(body));
                        let helper_body = match *body {
                            "sort_chars" => format!(
                                "fn {helper_name}(w: string) -> string {{\n    return sortchars(w);\n}}\n"
                            ),
                            b => format!(
                                "fn {helper_name}(w: string) -> string {{\n    return {b};\n}}\n"
                            ),
                        };
                        let mut helpers = e.helpers.clone();
                        helpers.push(MAPWORDS_TEMPLATE.replace("NAME", &helper_name));
                        helpers.push(helper_body);
                        if *body == "sort_chars" {
                            helpers.push(SORTCHARS_FN.to_string());
                        }
                        fresh.push(Expr {
                            outs,
                            mog: format!("map_{helper_name}({})", e.mog),
                            helpers,
                            depth,
                        });
                    }
                    // map(library op) over the word list — hole power via exec.
                    for &(op_name, op_mog) in &lib_str_ops {
                        let mut outs = Vec::with_capacity(e.outs.len());
                        let mut ok = true;
                        for v in &e.outs {
                            let V::L(l) = v else { unreachable!() };
                            let mut mapped = Vec::with_capacity(l.len());
                            for w in l {
                                match apply_lib(op_mog, op_name, w) {
                                    Some(o) => mapped.push(o),
                                    None => {
                                        ok = false;
                                        break;
                                    }
                                }
                            }
                            if !ok {
                                break;
                            }
                            outs.push(V::L(mapped));
                        }
                        if !ok {
                            continue;
                        }
                        let mut helpers = e.helpers.clone();
                        helpers.push(op_mog.to_string());
                        helpers.push(MAPWORDS_TEMPLATE.replace("NAME", op_name));
                        fresh.push(Expr { outs, mog: format!("map_{op_name}({})", e.mog), helpers, depth });
                    }
                    // SORT the word list (alpha asc / by length asc) — bounded
                    // by the node budget so it cannot blow up the search.
                    for (k_name, alpha) in [("alpha", true), ("len", false)] {
                        let outs: Vec<V> = e
                            .outs
                            .iter()
                            .map(|v| {
                                let V::L(l) = v else { unreachable!() };
                                let mut l = l.clone();
                                if alpha {
                                    l.sort();
                                } else {
                                    l.sort_by_key(|w| w.chars().count());
                                }
                                V::L(l)
                            })
                            .collect();
                        let cmp = if alpha { "ws[j] < ws[m]" } else { "ws[j].len < ws[m].len" };
                        let mut helpers = e.helpers.clone();
                        helpers.push(format!(
                            "fn sortw_{k_name}(xs: [string]) -> [string] {{\n    ws: [string] = [];\n    for e in xs {{\n        ws.push(e);\n    }}\n    i: i64 = 0;\n    while i < ws.len {{\n        m: i64 = i;\n        j: i64 = i + 1;\n        while j < ws.len {{\n            if {cmp} {{\n                m = j;\n            }}\n            j = j + 1;\n        }}\n        t: string = ws[i];\n        ws[i] = ws[m];\n        ws[m] = t;\n        i = i + 1;\n    }}\n    return ws;\n}}\n"
                        ));
                        fresh.push(Expr { outs, mog: format!("sortw_{k_name}({})", e.mog), helpers, depth });
                    }
                    // join with " " -> Str.
                    let outs: Vec<V> = e
                        .outs
                        .iter()
                        .map(|v| {
                            let V::L(l) = v else { unreachable!() };
                            V::S(l.join(" "))
                        })
                        .collect();
                    let mut helpers = e.helpers.clone();
                    helpers.push(JOINWORDS_FN.to_string());
                    fresh.push(Expr {
                        outs,
                        mog: format!("joinwords({})", e.mog),
                        helpers,
                        depth,
                    });
                }
                V::I(_) => {}
            }
        }

        // Winner (exact match) is pushed to the pool DIRECTLY, bypassing the
        // dedup/prune so best-first can never discard it.
        if let Some(win) = fresh.iter().find(|e| e.outs == expected).cloned() {
            pool.push(win);
        } else {
            // GUIDED best-first: keep only the top-K fresh expressions by goal
            // similarity to the target (char-multiset overlap of each rendered
            // output with the expected string). This is what makes hole power
            // affordable — 5-core + N-library ops fan out wide, but only the most
            // promising survive to expand, so depth-4 compositions (anti_shuffle)
            // stay reachable within the node budget. Emergent: the DATA (distance
            // to target), not a hand op-list, decides which branches live.
            fresh.retain(|e| seen.insert(e.outs.clone()));
            fresh.sort_by(|a, b| {
                goal_score(&b.outs, &expected_str).cmp(&goal_score(&a.outs, &expected_str))
            });
            fresh.truncate(MAX_POOL_PER_TYPE);
            for e in fresh {
                pool.push(e);
            }
        }

        // Accept: any pool expression whose outputs equal the expected vector.
        if let Some(win) = pool.iter().find(|e| e.outs == expected) {
            let mut helpers: Vec<String> = Vec::new();
            let mut seen_h: HashSet<String> = HashSet::new();
            for h in &win.helpers {
                if seen_h.insert(h.clone()) {
                    helpers.push(h.clone());
                }
            }
            let ret_ty = match expected[0] {
                V::I(_) => "i64",
                _ => "string",
            };
            let code = format!(
                "fn {name}(s: string) -> {ret_ty} {{\n    return {};\n}}\n\n{}",
                win.mog,
                helpers.join("\n")
            );
            if crate::runtime::code_reproduces_examples(&code, &problem.examples) {
                return Some(SolveResult {
                    success: true,
                    code,
                    method: "typed-enum-str".to_string(),
                    error: None,
                    metadata: Default::default(),
                });
            }
        }
    }
    None
}

/// Sort the chars of a string — emitted once per program when used.
const SORTCHARS_FN: &str = "fn sortchars(w: string) -> string {\n    cs: [string] = [];\n    for ch in w {\n        cs.push(ch);\n    }\n    i: i64 = 0;\n    while i < cs.len {\n        m: i64 = i;\n        j: i64 = i + 1;\n        while j < cs.len {\n            if cs[j] < cs[m] {\n                m = j;\n            }\n            j = j + 1;\n        }\n        t: string = cs[i];\n        cs[i] = cs[m];\n        cs[m] = t;\n        i = i + 1;\n    }\n    out: string = \"\";\n    for c in cs {\n        out = out + c;\n    }\n    return out;\n}\n";

/// Join a word list with single spaces.
const JOINWORDS_FN: &str = "fn joinwords(ws: [string]) -> string {\n    out: string = \"\";\n    i: i64 = 0;\n    while i < ws.len {\n        out = out + ws[i];\n        if i < ws.len - 1 {\n            out = out + \" \";\n        }\n        i = i + 1;\n    }\n    return out;\n}\n";

/// Map a helper over a word list. NAME is substituted per element fn.
const MAPWORDS_TEMPLATE: &str = "fn map_NAME(ws: [string]) -> [string] {\n    out: [string] = [];\n    for w in ws {\n        out.push(NAME(w));\n    }\n    return out;\n}\n";

/// True if the op's signature line returns the given type (`-> string`/`-> i64`).
fn op_returns(mog: &str, ty: &str) -> bool {
    mog.lines()
        .next()
        .map(|l| l.replace(' ', "").contains(&format!("->{ty}")))
        .unwrap_or(false)
}

/// True if the op's single parameter is a `string` (so it composes on Str exprs).
fn op_takes_string(mog: &str) -> bool {
    let Some(open) = mog.find('(') else { return false };
    let Some(close) = mog[open..].find(')') else { return false };
    let params = &mog[open + 1..open + close];
    params.replace(' ', "").contains(":string")
}

/// Goal-similarity score: summed char-MULTISET overlap of each rendered output
/// with its expected string. Higher = closer to the target; a list renders as
/// its space-join; non-string expected examples score 0 (they steer via exact
/// match, not this heuristic).
fn goal_score(outs: &[V], expected_str: &[String]) -> usize {
    outs.iter()
        .zip(expected_str.iter())
        .map(|(o, exp)| {
            if exp.is_empty() {
                return 0;
            }
            let s = match o {
                V::S(s) => s.clone(),
                V::L(l) => l.join(" "),
                V::I(_) => return 0,
            };
            char_multiset_overlap(&s, exp)
        })
        .sum()
}

/// Size of the char multiset intersection of two strings.
fn char_multiset_overlap(a: &str, b: &str) -> usize {
    use std::collections::HashMap;
    let mut counts: HashMap<char, i32> = HashMap::new();
    for c in a.chars() {
        *counts.entry(c).or_insert(0) += 1;
    }
    let mut shared = 0usize;
    for c in b.chars() {
        let e = counts.entry(c).or_insert(0);
        if *e > 0 {
            *e -= 1;
            shared += 1;
        }
    }
    shared
}

#[allow(dead_code)]
fn hash8(s: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    format!("{:x}", h.finish() & 0xffff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::Example;

    fn prob(rows: &[(&str, &str)]) -> Problem {
        Problem {
            name: "t".to_string(),
            examples: rows
                .iter()
                .map(|(i, o)| Example {
                    inputs: vec![Value::Str(i.to_string())],
                    expected: Value::Str(o.to_string()),
                })
                .collect(),
            ..Problem::default()
        }
    }

    #[test]
    fn enumerates_anti_shuffle_shape() {
        // join(map(split(s, " "), sort_chars), " ") — the depth-3 second-order
        // composition no op or schema reaches.
        let p = prob(&[
            ("Hi", "Hi"),
            ("hello", "ehllo"),
            ("Hello World!!!", "Hello !!!Wdlor"),
        ]);
        let r = try_typed_enum_str(&p, "anti_shuffle").expect("must enumerate the composition");
        assert_eq!(r.method, "typed-enum-str");
        assert!(crate::runtime::code_reproduces_examples(&r.code, &p.examples));
    }

    #[test]
    fn refuses_off_space_data() {
        // Arbitrary unrelated outputs must not be fitted.
        let p = prob(&[("abc", "qqq"), ("de", "zz9"), ("f", "!!")]);
        assert!(try_typed_enum_str(&p, "f").is_none());
    }

    #[test]
    fn enumerates_word_sort_join() {
        // join(sortw_alpha(split(s, " ")), " ") — the bounded sort-words op.
        let p = prob(&[
            ("c a b", "a b c"),
            ("zzz aaa mmm", "aaa mmm zzz"),
            ("one two", "one two"),
        ]);
        let r = try_typed_enum_str(&p, "sortwords").expect("must enumerate word-sort");
        assert!(crate::runtime::code_reproduces_examples(&r.code, &p.examples));
    }
}
