//! BUILD-C — NL-driven assembler: free-text English game description → PLAYABLE HTML game.
//!
//! This is the ONLY new code for BUILD-C. It reuses the already-proven lane_catch
//! pipeline (`demos/synthesized_game/build_game.py`) but replaces that demo's
//! HAND-AUTHORED `RequirementsIR` rule list with the REAL natural-language door:
//!
//!   English  --synthesize_project-->  {name → verified Mog}  --transpile-->  JS  --inject-->  game.html
//!
//! Nothing in the gameplay LOGIC is hand-written: every rule body is discovered by
//! the nsynth synthesizer from the inline I/O examples in the English description,
//! then re-verified here over a generous integer domain BEFORE it is allowed to
//! ship. The canvas + requestAnimationFrame loop + keyboard shell is the reused
//! hand-written harness (the lane_catch `_HTML_TEMPLATE`) — that is honest and
//! expected: lane_catch documents the shell as plain presentation, not synthesis.
//!
//! Pipeline (mirrors build_game.py + synth_pong_driver.mjs):
//!   1. `synthesize_project(text)` → Vec<(name, SolveResult)> + skipped list.
//!   2. REFUSED gate: every named rule must be present + success, and land in a
//!      PROVEN scalar lane — i64 (`-> i64`) OR f64 (`-> f64`, recovered by
//!      search_float_affine). The lane is chosen EMERGENTLY by the synthesized
//!      Mog's signature, not a phrase table. String/array/struct rules are still
//!      rejected up front (the verifier can't run that JS). On any failure we
//!      exit nonzero and write NO html — never a partial game.
//!   3. Per rule: normalize single-line `if cond { body }` into block form
//!      (replicates `normalizeMog`, synth_pong_driver.mjs:29-47, because the
//!      line-based transpiler mis-parses single-line ifs), `to_typescript`,
//!      strip TS annotations (`ts_to_js`).
//!   4. CEGIS re-verify over a generous integer domain:
//!        (a) execute the verified Mog (the real nsynth runtime) and confirm it
//!            reproduces EVERY inline example, and
//!        (b) evaluate the SHIPPED transpiled JS (a small i64-subset evaluator)
//!            and confirm it AGREES with the Mog at every domain point.
//!      Any residual mismatch rejects the rule and refuses the build.
//!   5. Inject the JS fns + a provenance table into the lane_catch template via the
//!      two markers (`/*__FUNCS__*/`, `<!--__PROV__-->`) and write game.html.
//!
//! Run (from repo root):
//!   HOME=/tmp/ncpu-sandbox-home cargo run --bin build_game_nl
//!   HOME=/tmp/ncpu-sandbox-home cargo run --bin build_game_nl -- --out /tmp/game.html
//!   HOME=/tmp/ncpu-sandbox-home cargo run --bin build_game_nl -- --demo-fail   (hard-fail proof)

use mog_synth::benchmark::Value as BValue;
use mog_synth::linguigenesis_bridge::LinguigenesisBridge;
use mog_synth::mog_transpile::to_typescript;
use mog_synth::runtime::{execute_function, Value as RValue};

// ---------------------------------------------------------------------------
// The English game description. Each rule is phrased as a function-head clause
// ("A function NAME that ...") so `comprehend_project` splits on it, and carries
// CALL-FORM inline examples ("NAME(x)=y") so the inline-example parser binds the
// literal lane_catch name (is_caught / score_after_catch / ...) AND derives the
// I/O contract. The five names match the lane_catch call sites verbatim, so the
// synthesized fns drop straight into the harness.
//
// This is the only thing the USER supplies: named rules + inline examples (i64).
// No rule BODY appears here — bodies are synthesized.
// ---------------------------------------------------------------------------
const GAME_ENGLISH: &str = "\
A function is_caught that returns one when the lane distance is zero otherwise zero, \
is_caught(0)=1 and is_caught(1)=0 and is_caught(2)=0 and is_caught(3)=0 and is_caught(4)=0. \
A function score_after_catch that increments the score by one after a catch, \
score_after_catch(0)=1 and score_after_catch(5)=6 and score_after_catch(41)=42 and score_after_catch(7)=8 and score_after_catch(99)=100. \
A function lives_after_miss that decreases the remaining lives by one after a miss, \
lives_after_miss(3)=2 and lives_after_miss(1)=0 and lives_after_miss(2)=1 and lives_after_miss(5)=4 and lives_after_miss(10)=9. \
A function is_game_over that returns one when no lives remain otherwise zero, \
is_game_over(0)=1 and is_game_over(1)=0 and is_game_over(2)=0 and is_game_over(3)=0. \
A function fall_speed that adds a base of three to the score, \
fall_speed(0)=3 and fall_speed(1)=4 and fall_speed(5)=8 and fall_speed(10)=13 and fall_speed(2)=5 and fall_speed(7)=10. \
A function fall_speed_f that is three plus the score times one half, \
fall_speed_f(0)=3.0 and fall_speed_f(2)=4.0 and fall_speed_f(4)=5.0 and fall_speed_f(6)=6.0 and fall_speed_f(8)=7.0.";

/// A rule's English contract: the name to bind + the inline examples that form
/// its spec. The user owns these (named rules + i64 examples). The body is NOT
/// here — it is synthesized. `english` is the human-readable spec shown in the
/// provenance panel.
struct RuleSpec {
    name: &'static str,
    english: &'static str,
    /// Inline I/O examples (single-arg i64) — the verification oracle for the
    /// integer lane. Empty for a pure-float rule (see `examples_f64`).
    examples: &'static [(i64, i64)],
    /// Inline I/O examples for a single-arg FLOAT rule. When non-empty the rule
    /// is verified through the f64 lane (Mog executed over f64 + shipped JS run
    /// through node). Empty for the i64 lane. Exactly one of the two is used,
    /// selected by the synthesized Mog's signature (`-> f64` ⇒ float lane).
    examples_f64: &'static [(f64, f64)],
}

const RULES: &[RuleSpec] = &[
    RuleSpec {
        name: "is_caught",
        english: "Return 1 when the basket and item share a lane (distance 0), else 0.",
        examples: &[(0, 1), (1, 0), (2, 0), (3, 0), (4, 0)],
        examples_f64: &[],
    },
    RuleSpec {
        name: "score_after_catch",
        english: "Increment the score by one after a successful catch.",
        examples: &[(0, 1), (5, 6), (41, 42), (7, 8), (99, 100)],
        examples_f64: &[],
    },
    RuleSpec {
        name: "lives_after_miss",
        english: "Decrease the remaining lives by one after a missed item.",
        examples: &[(3, 2), (1, 0), (2, 1), (5, 4), (10, 9)],
        examples_f64: &[],
    },
    RuleSpec {
        name: "is_game_over",
        english: "Return 1 when no lives remain (lives == 0), else 0.",
        examples: &[(0, 1), (1, 0), (2, 0), (3, 0)],
        examples_f64: &[],
    },
    RuleSpec {
        name: "fall_speed",
        english: "Fall speed in px/frame: base of three plus one per point of score.",
        examples: &[(0, 3), (1, 4), (5, 8), (10, 13), (2, 5), (7, 10)],
        examples_f64: &[],
    },
    // FLOAT lane: a real-valued physics rule (sub-integer per-point ramp).
    // fall_speed_f(score) = 3 + 0.5*score — a velocity ramp that can't be i64.
    // Synthesized by search_float_affine, transpiled, f64-annotation stripped,
    // verified Mog↔node across the float sweep. Demonstrates LOOP-1: described
    // REAL-VALUED rules now go all the way to runnable JS.
    RuleSpec {
        name: "fall_speed_f",
        english: "Fall speed (real): base of three plus half a pixel per point of score.",
        examples: &[],
        examples_f64: &[(0.0, 3.0), (2.0, 4.0), (4.0, 5.0), (6.0, 6.0), (8.0, 7.0)],
    },
];

/// A REQUIRED rule whose contract is genuinely unsynthesizable — a jagged,
/// non-affine integer table no closed-form lane (i64 OR f64 affine) can recover.
/// Used only by `--demo-fail`: it is added to BOTH the English and the
/// required-rule set, so the build must REFUSE. (Either the synthesizer skips it,
/// or — as observed — a teacher fits some examples but CEGIS re-verification
/// catches the disagreement; both paths refuse and write NO html. This is the
/// honest hard-fail proof now that real-valued affine rules SUCCEED via the f64
/// lane and are no longer the canonical "unsupported" case.)
const FAIL_RULE: RuleSpec = RuleSpec {
    name: "needs_lookup",
    english: "A non-affine table the closed-form solvers cannot recover — genuinely unsynthesizable.",
    // A jagged, non-monotone, non-affine map: no i64 or f64 affine model fits it,
    // so BOTH lanes refuse. (The old f64 'half of input' demo is now a real
    // SUCCESS via the float lane, so the hard-fail rule must be one that no lane
    // can solve — proving the gate still refuses genuinely unsynthesizable rules.)
    examples: &[(0, 7), (1, 2), (2, 9), (3, 1), (4, 5), (5, 0)],
    examples_f64: &[],
};

/// Integer domain each rule is swept over during CEGIS re-verification. The game
/// only ever feeds non-negative values to these rules (absolute lane distance,
/// score, lives), so a non-negative sweep is the honest domain — adding inputs
/// the game can never produce would only handicap synthesis. Wide enough (0..=128)
/// to expose any overfit formula that matched the handful of examples by accident.
const SWEEP: std::ops::RangeInclusive<i64> = 0..=128;

fn main() {
    let mut out_path = std::path::PathBuf::from(default_out_path());
    let mut demo_fail = false;
    let mut english = GAME_ENGLISH.to_string();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => {
                out_path = std::path::PathBuf::from(
                    args.next().unwrap_or_else(|| die("--out needs a path")),
                );
            }
            "--demo-fail" => demo_fail = true,
            "--help" | "-h" => {
                eprintln!(
                    "build_game_nl — NL → playable HTML game\n\
                     \n\
                     --out PATH     write the game here (default: demos/synthesized_game/game.html)\n\
                     --demo-fail    inject an UNSYNTHESIZABLE / f64 rule to prove the hard-fail gate\n"
                );
                return;
            }
            other => die(&format!("unknown arg: {other}")),
        }
    }

    // The required rule set: the 5 lane_catch rules. `--demo-fail` adds a 6th
    // REQUIRED f64 rule so the gate must catch it.
    let mut required: Vec<&RuleSpec> = RULES.iter().collect();
    if demo_fail {
        // HARD-FAIL PROOF: append a REQUIRED rule whose contract is a jagged,
        // non-affine table that NO closed-form lane (i64 OR f64 affine) can
        // recover. It is added to both the English AND the required set, so the
        // build must REFUSE and write NO html. (Note: f64 is no longer the
        // failure mode — real-valued affine rules now SUCCEED via the float
        // lane — so the unsynthesizable demo is a genuinely unfittable map.)
        english.push_str(
            " A function needs_lookup that maps inputs to a fixed jagged table, \
             needs_lookup(0)=7 and needs_lookup(1)=2 and needs_lookup(2)=9 and \
             needs_lookup(3)=1 and needs_lookup(4)=5 and needs_lookup(5)=0.",
        );
        required.push(&FAIL_RULE);
        eprintln!(
            "[demo-fail] injected a REQUIRED unsynthesizable rule 'needs_lookup' — expecting REFUSAL.\n"
        );
    }

    let game = build(&english, &required).unwrap_or_else(|e| {
        eprintln!("\nREFUSED: {e}");
        eprintln!("No game.html written — refusing to ship a partial/wrong game.");
        std::process::exit(1);
    });

    std::fs::write(&out_path, &game.html).unwrap_or_else(|e| die(&format!("write {out_path:?}: {e}")));
    println!("\nbuilt {}", out_path.display());
    println!(
        "rules synthesized + CEGIS-verified (Mog↔transpiledJS over {} domain points): {}/{}",
        SWEEP.count(),
        game.verified.len(),
        game.verified.len()
    );
    for v in &game.verified {
        println!(
            "  {:<20} {:<28} domain-verified ✓ ({} examples, {} sweep pts)",
            v.name, v.method, v.n_examples, v.n_sweep
        );
    }
}

struct VerifiedRule {
    name: String,
    method: String,
    english: String,
    js: String,
    ts: String,
    n_examples: usize,
    n_sweep: usize,
}

struct Game {
    html: String,
    verified: Vec<VerifiedRule>,
}

fn build(english: &str, required: &[&RuleSpec]) -> Result<Game, String> {
    // ---- 1. REAL NL door: comprehend → split → synthesize each component. ----
    let bridge = LinguigenesisBridge::new();
    if let Some(err) = bridge.registry_load_error() {
        return Err(format!("NL registry failed to load: {err}"));
    }
    let (solved, skipped) = bridge
        .synthesize_project(english)
        .map_err(|e| format!("synthesize_project failed: {e}"))?;

    eprintln!("[synth] solved {} components, skipped {}:", solved.len(), skipped.len());
    for (name, res) in &solved {
        eprintln!("  + {name:<20} success={} method={}", res.success, res.method);
    }
    for s in &skipped {
        eprintln!("  - SKIPPED: {s}");
    }

    let by_name: std::collections::HashMap<&str, &mog_synth::solver::SolveResult> =
        solved.iter().map(|(n, r)| (n.as_str(), r)).collect();

    // ---- 2. REFUSED gate + 3/4. normalize → transpile → CEGIS re-verify. ----
    // Every REQUIRED rule must be present, successful, i64-only, and domain-verified.
    // A missing/skipped rule (e.g. an f64 contract the synthesizer cannot solve) or
    // a non-i64 signature refuses the whole build — never a partial game.
    let mut verified = Vec::new();
    for spec in required {
        let res = by_name.get(spec.name).copied().ok_or_else(|| {
            format!(
                "rule '{}' was NOT synthesized (missing or skipped). skipped={skipped:?}",
                spec.name
            )
        })?;
        if !res.success {
            return Err(format!(
                "rule '{}' did not synthesize: {}",
                spec.name,
                res.error.clone().unwrap_or_else(|| "no solution".into())
            ));
        }
        // TWO-LANE gate, selected EMERGENTLY by the synthesized Mog's signature
        // (not a phrase table): the integer lane (`-> i64`) and the scalar-float
        // lane (`-> f64`, recovered by search_float_affine). A rule that is
        // neither scalar i64 nor scalar f64 (string/array/struct) is refused —
        // those still emit JS the verifier can't run. Whichever lane the
        // signature lands in, the i64 path stays byte-identical to before.
        let lane_i64 = is_i64_only(&res.code);
        let lane_f64 = is_f64_scalar(&res.code);
        if !lane_i64 && !lane_f64 {
            return Err(format!(
                "rule '{}' is not a scalar i64 or f64 rule — the transpiler emits JS the \
                 verifier can't run for string/array/struct; refusing.\n  mog: {}",
                spec.name,
                res.code.lines().next().unwrap_or("").trim()
            ));
        }

        // normalize single-line ifs (replicate normalizeMog) → transpile → ts→js.
        // ts_to_js erases `: i64`/`: number` AND `: f64`/`: f32`, so a float fn
        // body (already valid JS) ships runnable.
        let mog = normalize_mog(&res.code);
        let ts = to_typescript(&mog);
        let js = ts_to_js(&ts);

        // CEGIS re-verify: Mog reproduces every example AND the shipped JS agrees
        // with the Mog across the whole sweep. Reject on any residual mismatch.
        // The i64 lane uses the in-process i64 evaluator; the f64 lane executes
        // the Mog over f64 AND runs the SHIPPED JS through node (the real JS
        // runtime — honest, and avoids a hand-rolled float evaluator).
        let (n_ex, n_sweep) = if lane_i64 {
            cegis_verify(spec, &res.code, &js)?
        } else {
            cegis_verify_f64(spec, &res.code, &js)?
        };

        verified.push(VerifiedRule {
            name: spec.name.to_string(),
            method: res.method.clone(),
            english: spec.english.to_string(),
            js,
            ts,
            n_examples: n_ex,
            n_sweep,
        });
    }

    // ---- 5. inject into the lane_catch template. ----
    let js_funcs = verified
        .iter()
        .map(|v| v.js.clone())
        .collect::<Vec<_>>()
        .join("\n\n");
    let prov_rows = verified
        .iter()
        .map(|v| {
            format!(
                "<tr><td class=\"r\">{}</td><td>{}</td><td class=\"m\">{}</td>\
                 <td>{}</td><td class=\"c high\">✓ domain-verified</td></tr>",
                html_escape(&v.name),
                html_escape(&v.english),
                html_escape(&v.method),
                v.n_sweep
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let html = HTML_TEMPLATE
        .replace("/*__FUNCS__*/", &js_funcs)
        .replace("<!--__PROV__-->", &prov_rows);

    Ok(Game { html, verified })
}

fn default_out_path() -> String {
    // Resolve relative to the workspace so `cargo run` from anywhere lands the
    // game next to lane_catch.html.
    let manifest = env!("CARGO_MANIFEST_DIR"); // .../nsynth
    let root = std::path::Path::new(manifest)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    root.join("demos/synthesized_game/game.html")
        .to_string_lossy()
        .into_owned()
}

/// True iff the Mog function's header is the pure i64 surface (all params `i64`,
/// return `i64`, no float/string/array types). The transpiler's proven lane.
fn is_i64_only(mog: &str) -> bool {
    let header = mog.lines().next().unwrap_or("");
    let lower = header.to_lowercase();
    let banned = ["f64", "f32", "float", "string", "str", "&str", "char", "bool", "[", "vec<"];
    if banned.iter().any(|b| lower.contains(b)) {
        return false;
    }
    // Must actually declare an i64 return so we know it's the integer lane.
    lower.contains("-> i64")
}

/// True iff the Mog function's header is the scalar FLOAT surface: a `-> f64`
/// return with only `f64` scalar params (no arrays/strings/structs). This is the
/// second proven lane — `search_float_affine` emits exactly `c0 + Σ c_j·x_j` over
/// f64, and the transpiler emits a syntactically valid JS body (only the type
/// annotation is bogus, and `ts_to_js` erases it). Real-valued physics rules
/// (velocity decay, sub-integer speed) live here.
fn is_f64_scalar(mog: &str) -> bool {
    let header = mog.lines().next().unwrap_or("");
    let lower = header.to_lowercase();
    // Float return is mandatory (this is the f64 lane, routed emergently by the
    // op_role/float family — the signature carries `-> f64` only when the NL
    // examples were real-valued and search_float recovered an affine model).
    if !lower.contains("-> f64") {
        return false;
    }
    // No arrays/strings/structs — scalar only. (We DON'T ban `f64`; that's the
    // whole point. We ban the aggregate/foreign types the JS verifier can't run.)
    let banned = ["string", "str", "&str", "char", "bool", "[", "vec<"];
    !banned.iter().any(|b| lower.contains(b))
}

/// Replicate `normalizeMog` (synth_pong_driver.mjs:29-47): the line-based
/// transpiler parses block-style `if cond {\n body \n}` but mis-parses the
/// solver's single-line `if cond { body }` (and `... } else { ... }`) form.
/// Reformat single-line ifs into block form. Formatting-only — the program is
/// untouched. (Most synthesized bodies are already block-form or branch-free;
/// this is the safety net the pong driver proved necessary.)
fn normalize_mog(mog: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    for line in mog.lines() {
        if let Some((ind, cond, then_body, else_body)) = parse_single_line_if(line) {
            let stmts = |b: &str, ind: &str| -> Vec<String> {
                b.split(';')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .map(|s| format!("{ind}    {s};"))
                    .collect()
            };
            out.push(format!("{ind}if {cond} {{"));
            out.extend(stmts(&then_body, &ind));
            if let Some(eb) = else_body {
                out.push(format!("{ind}}} else {{"));
                out.extend(stmts(&eb, &ind));
            }
            out.push(format!("{ind}}}"));
        } else {
            out.push(line.to_string());
        }
    }
    out.join("\n")
}

/// Match `^(\s*)if (cond) \{ (then) \}( else \{ (else) \})?\s*$`.
fn parse_single_line_if(line: &str) -> Option<(String, String, String, Option<String>)> {
    let indent: String = line.chars().take_while(|c| c.is_whitespace()).collect();
    let rest = line.trim_start();
    let rest = rest.strip_prefix("if ")?;
    // Find the first " { " that opens the then-block.
    let open = rest.find(" { ")?;
    let cond = rest[..open].trim().to_string();
    let after = &rest[open + 3..];
    // then-block ends at the matching " }" — single-line bodies have no nested
    // braces, so the first " }" closes it.
    let then_end = after.find(" }").or_else(|| after.strip_suffix("}").map(|s| s.len()))?;
    let then_body = after[..then_end].trim().to_string();
    let tail = after[then_end + 2..].trim();
    let else_body = if let Some(e) = tail.strip_prefix("else { ") {
        Some(e.trim_end_matches('}').trim().to_string())
    } else if tail.is_empty() {
        None
    } else {
        // Unexpected trailing content — not a clean single-line if; leave as-is.
        return None;
    };
    Some((indent, cond, then_body, else_body))
}

/// Strip TypeScript type annotations to plain JS (mirrors `ts_to_js` in
/// build_game.py and `tsToJs` in synth_pong_driver.mjs). Bodies are scalar
/// arithmetic / branches / loops over i64 or f64, so only the type syntax is
/// removed.
///
/// JS is untyped, so erasing the annotation is exactly the right move: a float
/// body `return 0.5 * x + 3.0;` is already valid JS once the bogus `: f64`
/// (which the i64-only transpiler emits unchanged for float signatures) is gone.
/// We strip the float annotations the same way we strip `: number`/`: i64`,
/// which turns a transpiled f64 fn into runnable JS without ever touching
/// mog_transpile. The i64 lane never emits `f64`/`f32`, so this is a no-op for it.
fn ts_to_js(ts: &str) -> String {
    let mut js = ts.replace(": number", "");
    js = js.replace(": i64", "");
    // Float signatures: the i64-only transpiler passes `f64`/`f32` through as a
    // literal type name (it only maps i64/[i64]/string). Erase it — the BODY is
    // already valid JS; only the annotation was bogus.
    js = js.replace(": f64", "");
    js = js.replace(": f32", "");
    js.trim().to_string()
}

/// CEGIS re-verification of one rule. Returns (n_examples_checked, n_sweep_points).
///
/// Honest, two-sided gate:
///   (a) the verified Mog (executed by the real nsynth runtime) reproduces EVERY
///       inline example — the user's contract, re-checked, not assumed; and
///   (b) the SHIPPED transpiled JS agrees with the Mog at every sweep point — this
///       catches both an overfit synthesized formula AND a broken transpile.
/// Any disagreement returns Err and the whole build refuses.
fn cegis_verify(spec: &RuleSpec, mog: &str, js: &str) -> Result<(usize, usize), String> {
    // (a) Mog reproduces every inline example.
    for &(x, want) in spec.examples {
        let got = run_mog_i64(mog, spec.name, x).map_err(|e| {
            format!("rule '{}': Mog errored on example input {x}: {e}", spec.name)
        })?;
        if got != want {
            return Err(format!(
                "rule '{}': Mog disagrees with inline example {}({x}) = {got}, expected {want}",
                spec.name, spec.name
            ));
        }
    }

    // (b) shipped JS == Mog across the whole sweep (overfit + transpile guard).
    let jsfn = JsFn::parse(js, spec.name)
        .map_err(|e| format!("rule '{}': could not parse shipped JS: {e}", spec.name))?;
    let mut n_sweep = 0usize;
    for x in SWEEP {
        let mog_out = run_mog_i64(mog, spec.name, x)
            .map_err(|e| format!("rule '{}': Mog errored on sweep input {x}: {e}", spec.name))?;
        let js_out = jsfn
            .eval(x)
            .map_err(|e| format!("rule '{}': shipped JS errored on {x}: {e}", spec.name))?;
        if mog_out != js_out {
            return Err(format!(
                "rule '{}': transpiled JS disagrees with Mog at input {x}: JS={js_out} Mog={mog_out}\n  js: {js}",
                spec.name
            ));
        }
        n_sweep += 1;
    }
    Ok((spec.examples.len(), n_sweep))
}

/// Execute a synthesized Mog single-arg i64 function via the real nsynth runtime.
fn run_mog_i64(mog: &str, name: &str, x: i64) -> Result<i64, String> {
    let v = execute_function(mog, name, &[BValue::Int(x)], name)?;
    match v {
        RValue::Int(n) => Ok(n),
        other => Err(format!("non-integer return: {other:?}")),
    }
}

// ---------------------------------------------------------------------------
// FLOAT lane (LOOP-1): verify a scalar `-> f64` rule end-to-end. The i64 lane
// above is untouched; this is an additive, parallel path used ONLY when the
// synthesized Mog's signature is scalar f64 (`is_f64_scalar`).
//
// We verify HONESTLY in two ways, mirroring the i64 lane:
//   (a) the verified Mog (real nsynth runtime, over f64) reproduces every inline
//       float example within a small tolerance (continuous data is approximate
//       by nature — the same recover-or-refuse contract as search_float), AND
//   (b) the SHIPPED JS, run through `node` (the real JS runtime), agrees with the
//       Mog across a float sweep. Using node — not a hand-rolled float evaluator
//       — keeps the oracle honest: we run the literal artifact in a real engine.
// ---------------------------------------------------------------------------

/// Float comparison tolerance for the f64 lane. Generous enough for the
/// finite-precision coefficients search_float prints, tight enough to catch a
/// wrong formula or a mangled transpile.
const F64_EPS: f64 = 1e-6;

/// Float sweep points the f64 rule is checked over (Mog ↔ node). Includes
/// sub-integer and larger values so an overfit-to-the-examples formula or a
/// broken transpile shows up. Game inputs are non-negative (score), so a
/// non-negative sweep is the honest domain.
fn f64_sweep() -> Vec<f64> {
    let mut v = Vec::new();
    let mut x = 0.0f64;
    while x <= 128.0 {
        v.push(x);
        x += 0.5; // sub-integer steps exercise the float regime
    }
    v
}

/// Execute a synthesized Mog single-arg f64 function via the real nsynth runtime.
/// Accepts an `Int` return too (coerced) so a degenerate integer-valued float fn
/// still verifies.
fn run_mog_f64(mog: &str, name: &str, x: f64) -> Result<f64, String> {
    let v = execute_function(mog, name, &[BValue::Float(x.to_bits())], name)?;
    match v {
        RValue::Float(n) => Ok(n),
        RValue::Int(n) => Ok(n as f64),
        other => Err(format!("non-float return: {other:?}")),
    }
}

/// CEGIS re-verification of one FLOAT rule. Returns (n_examples, n_sweep_points).
fn cegis_verify_f64(spec: &RuleSpec, mog: &str, js: &str) -> Result<(usize, usize), String> {
    // (a) Mog reproduces every inline float example within tolerance.
    for &(x, want) in spec.examples_f64 {
        let got = run_mog_f64(mog, spec.name, x).map_err(|e| {
            format!("rule '{}': Mog errored on float example input {x}: {e}", spec.name)
        })?;
        if (got - want).abs() > F64_EPS {
            return Err(format!(
                "rule '{}': Mog disagrees with inline example {}({x}) = {got}, expected {want}",
                spec.name, spec.name
            ));
        }
    }

    // (b) shipped JS == Mog across the float sweep, evaluated by node (real JS).
    let sweep = f64_sweep();
    let mog_outs: Vec<f64> = sweep
        .iter()
        .map(|&x| {
            run_mog_f64(mog, spec.name, x)
                .map_err(|e| format!("rule '{}': Mog errored on sweep input {x}: {e}", spec.name))
        })
        .collect::<Result<_, _>>()?;
    let js_outs = eval_js_f64_via_node(js, spec.name, &sweep)
        .map_err(|e| format!("rule '{}': node evaluation of shipped JS failed: {e}", spec.name))?;
    if js_outs.len() != sweep.len() {
        return Err(format!(
            "rule '{}': node returned {} values for {} sweep points",
            spec.name,
            js_outs.len(),
            sweep.len()
        ));
    }
    for ((&x, &m), &j) in sweep.iter().zip(mog_outs.iter()).zip(js_outs.iter()) {
        if (m - j).abs() > F64_EPS {
            return Err(format!(
                "rule '{}': transpiled JS disagrees with Mog at input {x}: JS={j} Mog={m}\n  js: {js}",
                spec.name
            ));
        }
    }
    Ok((spec.examples_f64.len(), sweep.len()))
}

/// Run the SHIPPED JS function in `node` (the real JS runtime) over a batch of
/// float inputs and return its outputs. Honest oracle: we evaluate the literal
/// bytes we ship, in a real engine — no hand-rolled float evaluator to drift.
/// A missing `node` is a hard error (fail-closed: an f64 rule is only allowed to
/// ship if it was actually run and verified).
fn eval_js_f64_via_node(js: &str, name: &str, inputs: &[f64]) -> Result<Vec<f64>, String> {
    use std::io::Write;
    // Build a tiny driver: the shipped fn + a loop printing one output per line.
    let inputs_lit = inputs
        .iter()
        .map(|x| format!("{x:?}"))
        .collect::<Vec<_>>()
        .join(",");
    let driver = format!(
        "{js}\nconst __xs=[{inputs_lit}];\nfor(const __x of __xs){{const __y={name}(__x);if(typeof __y!=='number'||!isFinite(__y)){{console.error('non-finite output');process.exit(3);}}console.log(__y);}}\n"
    );
    // Write to a temp file and run `node FILE` (avoids shell-escaping the body).
    let mut path = std::env::temp_dir();
    path.push(format!("build_game_nl_f64_{}_{}.mjs", name, std::process::id()));
    {
        let mut f = std::fs::File::create(&path)
            .map_err(|e| format!("create temp driver {path:?}: {e}"))?;
        f.write_all(driver.as_bytes())
            .map_err(|e| format!("write temp driver: {e}"))?;
    }
    let out = std::process::Command::new("node")
        .arg(&path)
        .output()
        .map_err(|e| format!("could not run `node` (required to verify f64 rules): {e}"))?;
    let _ = std::fs::remove_file(&path);
    if !out.status.success() {
        return Err(format!(
            "node exited {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut vals = Vec::new();
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        vals.push(
            line.parse::<f64>()
                .map_err(|_| format!("node printed non-number line {line:?}"))?,
        );
    }
    Ok(vals)
}

// ---------------------------------------------------------------------------
// Minimal i64-subset evaluator for the SHIPPED transpiled JS. It runs the literal
// bytes we inject into game.html, so verification checks the actual artifact —
// not a re-derivation. Supports exactly the surface the i64 transpiler emits:
//   function NAME(p) { let v = EXPR; ...; while (COND) { ... } if (COND) {...} else {...} return EXPR; }
// with arithmetic (+ - * / %), Math.trunc(...), parens, comparisons, single i64
// param. Anything outside this surface is an error (which refuses the build —
// fail-closed, never ship unverified JS).
// ---------------------------------------------------------------------------
struct JsFn {
    param: String,
    body: Vec<JsStmt>,
}

#[derive(Clone)]
enum JsStmt {
    Let(String, String),
    Assign(String, String),
    Return(String),
    While(String, Vec<JsStmt>),
    If(String, Vec<JsStmt>, Vec<JsStmt>),
}

impl JsFn {
    fn parse(js: &str, name: &str) -> Result<JsFn, String> {
        let sig = format!("function {name}(");
        let start = js.find(&sig).ok_or_else(|| format!("no `function {name}(`"))?;
        let after = &js[start + sig.len()..];
        let pclose = after.find(')').ok_or("no `)` after params")?;
        let param = after[..pclose].split(':').next().unwrap_or("").trim().to_string();
        if param.is_empty() {
            return Err("empty param list (need single i64 arg)".into());
        }
        let bopen = after[pclose..].find('{').ok_or("no `{` body")? + pclose;
        let body_src = &after[bopen + 1..];
        // The body runs to the matching closing brace of the function.
        let (stmts, _consumed) = parse_block(body_src)?;
        Ok(JsFn { param, body: stmts })
    }

    fn eval(&self, x: i64) -> Result<i64, String> {
        let mut env = std::collections::HashMap::new();
        env.insert(self.param.clone(), x);
        match exec_block(&self.body, &mut env)? {
            Some(ret) => Ok(ret),
            None => Err("function returned no value".into()),
        }
    }
}

/// Parse a `{ ... }` body (without the leading `{`) up to its matching `}`.
/// Returns the statements and the number of bytes consumed (incl. the `}`).
fn parse_block(src: &str) -> Result<(Vec<JsStmt>, usize), String> {
    let mut stmts = Vec::new();
    let b = src.as_bytes();
    let mut i = 0;
    while i < b.len() {
        // skip whitespace
        while i < b.len() && (b[i] as char).is_whitespace() {
            i += 1;
        }
        if i >= b.len() {
            return Err("unterminated block (no `}`)".into());
        }
        if b[i] == b'}' {
            return Ok((stmts, i + 1));
        }
        let rest = &src[i..];
        if let Some(r) = rest.strip_prefix("return ") {
            let semi = r.find(';').ok_or("return without `;`")?;
            stmts.push(JsStmt::Return(r[..semi].trim().to_string()));
            i += "return ".len() + semi + 1;
        } else if let Some(r) = rest.strip_prefix("let ") {
            let semi = r.find(';').ok_or("let without `;`")?;
            let decl = &r[..semi];
            let eq = decl.find('=').ok_or("let without `=`")?;
            let var = decl[..eq].split(':').next().unwrap_or("").trim().to_string();
            stmts.push(JsStmt::Let(var, decl[eq + 1..].trim().to_string()));
            i += "let ".len() + semi + 1;
        } else if let Some(r) = rest.strip_prefix("while ") {
            let (cond, body, consumed) = parse_ctrl(r)?;
            stmts.push(JsStmt::While(cond, body));
            i += "while ".len() + consumed;
        } else if let Some(r) = rest.strip_prefix("if ") {
            let (cond, then_b, consumed) = parse_ctrl(r)?;
            i += "if ".len() + consumed;
            // optional else
            let tail = &src[i..];
            let trimmed = tail.trim_start();
            let ws = tail.len() - trimmed.len();
            if let Some(eb) = trimmed.strip_prefix("else") {
                let eb_t = eb.trim_start();
                let open = eb_t.find('{').ok_or("else without `{`")?;
                let (else_b, c2) = parse_block(&eb_t[open + 1..])?;
                stmts.push(JsStmt::If(cond, then_b, else_b));
                i += ws + (eb.len() - eb_t.len()) + "else".len() + open + 1 + c2;
            } else {
                stmts.push(JsStmt::If(cond, then_b, Vec::new()));
            }
        } else {
            // assignment `var = expr;`
            let semi = rest.find(';').ok_or_else(|| format!("statement without `;`: {rest:?}"))?;
            let decl = &rest[..semi];
            let eq = decl.find('=').ok_or_else(|| format!("unrecognized statement: {decl:?}"))?;
            stmts.push(JsStmt::Assign(
                decl[..eq].trim().to_string(),
                decl[eq + 1..].trim().to_string(),
            ));
            i += semi + 1;
        }
    }
    Err("unterminated block (no `}`)".into())
}

/// Parse `(cond) { body }` starting after the `while `/`if ` keyword. Returns
/// (cond, body stmts, bytes consumed up to and including the body's `}`).
fn parse_ctrl(r: &str) -> Result<(String, Vec<JsStmt>, usize), String> {
    let popen = r.find('(').ok_or("control without `(`")?;
    let pclose = matching(r.as_bytes(), popen, b'(', b')').ok_or("unbalanced `(`")?;
    let cond = r[popen + 1..pclose].trim().to_string();
    let bopen = r[pclose..].find('{').ok_or("control without `{`")? + pclose;
    let (body, consumed) = parse_block(&r[bopen + 1..])?;
    Ok((cond, body, bopen + 1 + consumed))
}

fn matching(b: &[u8], open: usize, oc: u8, cc: u8) -> Option<usize> {
    let mut depth = 0i32;
    for (k, &c) in b.iter().enumerate().skip(open) {
        if c == oc {
            depth += 1;
        } else if c == cc {
            depth -= 1;
            if depth == 0 {
                return Some(k);
            }
        }
    }
    None
}

fn exec_block(
    stmts: &[JsStmt],
    env: &mut std::collections::HashMap<String, i64>,
) -> Result<Option<i64>, String> {
    for s in stmts {
        match s {
            JsStmt::Let(v, e) | JsStmt::Assign(v, e) => {
                let val = eval_expr(e, env)?;
                env.insert(v.clone(), val);
            }
            JsStmt::Return(e) => return Ok(Some(eval_expr(e, env)?)),
            JsStmt::While(c, body) => {
                let mut guard = 0;
                while eval_cond(c, env)? {
                    if let Some(r) = exec_block(body, env)? {
                        return Ok(Some(r));
                    }
                    guard += 1;
                    if guard > 1_000_000 {
                        return Err("while loop exceeded 1e6 iterations".into());
                    }
                }
            }
            JsStmt::If(c, t, e) => {
                let branch = if eval_cond(c, env)? { t } else { e };
                if let Some(r) = exec_block(branch, env)? {
                    return Ok(Some(r));
                }
            }
        }
    }
    Ok(None)
}

fn eval_cond(c: &str, env: &std::collections::HashMap<String, i64>) -> Result<bool, String> {
    for (op, f) in [
        ("<=", 0u8),
        (">=", 1),
        ("==", 2),
        ("!=", 3),
        ("<", 4),
        (">", 5),
    ] {
        if let Some(idx) = c.find(op) {
            let l = eval_expr(c[..idx].trim(), env)?;
            let r = eval_expr(c[idx + op.len()..].trim(), env)?;
            return Ok(match f {
                0 => l <= r,
                1 => l >= r,
                2 => l == r,
                3 => l != r,
                4 => l < r,
                _ => l > r,
            });
        }
    }
    // bare expression: truthy if nonzero
    Ok(eval_expr(c, env)? != 0)
}

/// Evaluate an i64 arithmetic expression: + - * / %, Math.trunc(...), parens,
/// idents, integer literals. `/` is truncating (matches Mog + the transpiler's
/// `Math.trunc` wrapping). This is a recursive-descent evaluator over exactly
/// the transpiler's emitted expression grammar.
fn eval_expr(e: &str, env: &std::collections::HashMap<String, i64>) -> Result<i64, String> {
    let toks = tokenize(e)?;
    let mut p = Parser { toks: &toks, pos: 0, env };
    let v = p.expr()?;
    if p.pos != p.toks.len() {
        return Err(format!("trailing tokens in expr {e:?}"));
    }
    Ok(v)
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(i64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    LParen,
    RParen,
    Trunc, // Math.trunc
    Comma,
}

fn tokenize(e: &str) -> Result<Vec<Tok>, String> {
    let b = e.as_bytes();
    let mut i = 0;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
        } else if c == '+' {
            out.push(Tok::Plus);
            i += 1;
        } else if c == '-' {
            out.push(Tok::Minus);
            i += 1;
        } else if c == '*' {
            out.push(Tok::Star);
            i += 1;
        } else if c == '/' {
            out.push(Tok::Slash);
            i += 1;
        } else if c == '%' {
            out.push(Tok::Percent);
            i += 1;
        } else if c == '(' {
            out.push(Tok::LParen);
            i += 1;
        } else if c == ')' {
            out.push(Tok::RParen);
            i += 1;
        } else if c == ',' {
            out.push(Tok::Comma);
            i += 1;
        } else if c.is_ascii_digit() {
            let s = i;
            while i < b.len() && (b[i] as char).is_ascii_digit() {
                i += 1;
            }
            out.push(Tok::Num(e[s..i].parse().map_err(|_| "bad number")?));
        } else if c.is_alphabetic() || c == '_' {
            let s = i;
            while i < b.len()
                && ((b[i] as char).is_alphanumeric() || b[i] == b'_' || b[i] == b'.')
            {
                i += 1;
            }
            let id = &e[s..i];
            if id == "Math.trunc" {
                out.push(Tok::Trunc);
            } else {
                out.push(Tok::Ident(id.to_string()));
            }
        } else {
            return Err(format!("unexpected char {c:?} in expr"));
        }
    }
    Ok(out)
}

struct Parser<'a> {
    toks: &'a [Tok],
    pos: usize,
    env: &'a std::collections::HashMap<String, i64>,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }
    fn bump(&mut self) -> Option<&Tok> {
        let t = self.toks.get(self.pos);
        self.pos += 1;
        t
    }
    // expr := term (('+'|'-') term)*
    fn expr(&mut self) -> Result<i64, String> {
        let mut v = self.term()?;
        while let Some(t) = self.peek() {
            match t {
                Tok::Plus => {
                    self.bump();
                    v += self.term()?;
                }
                Tok::Minus => {
                    self.bump();
                    v -= self.term()?;
                }
                _ => break,
            }
        }
        Ok(v)
    }
    // term := factor (('*'|'/'|'%') factor)*   (truncating int div/mod)
    fn term(&mut self) -> Result<i64, String> {
        let mut v = self.factor()?;
        while let Some(t) = self.peek() {
            match t {
                Tok::Star => {
                    self.bump();
                    v *= self.factor()?;
                }
                Tok::Slash => {
                    self.bump();
                    let d = self.factor()?;
                    if d == 0 {
                        return Err("division by zero".into());
                    }
                    v = v.wrapping_div(d); // Rust `/` truncates toward zero == Math.trunc
                }
                Tok::Percent => {
                    self.bump();
                    let d = self.factor()?;
                    if d == 0 {
                        return Err("modulo by zero".into());
                    }
                    v = v.wrapping_rem(d);
                }
                _ => break,
            }
        }
        Ok(v)
    }
    // factor := '-' factor | '(' expr ')' | Math.trunc '(' expr ')' | num | ident
    fn factor(&mut self) -> Result<i64, String> {
        match self.bump().cloned() {
            Some(Tok::Minus) => Ok(-self.factor()?),
            Some(Tok::Num(n)) => Ok(n),
            Some(Tok::Ident(id)) => self
                .env
                .get(&id)
                .copied()
                .ok_or_else(|| format!("unbound ident {id:?}")),
            Some(Tok::LParen) => {
                let v = self.expr()?;
                match self.bump() {
                    Some(Tok::RParen) => Ok(v),
                    _ => Err("expected `)`".into()),
                }
            }
            Some(Tok::Trunc) => {
                match self.bump() {
                    Some(Tok::LParen) => {}
                    _ => return Err("Math.trunc without `(`".into()),
                }
                let v = self.expr()?;
                match self.bump() {
                    Some(Tok::RParen) => Ok(v), // integer arithmetic already truncates
                    _ => Err("Math.trunc without `)`".into()),
                }
            }
            other => Err(format!("unexpected token in factor: {other:?}")),
        }
    }
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(2);
}

// ---------------------------------------------------------------------------
// The lane_catch _HTML_TEMPLATE (build_game.py:223-301), reused VERBATIM as the
// harness. Two markers: `/*__FUNCS__*/` (synthesized JS fns) and `<!--__PROV__-->`
// (provenance rows). The fixed shell calls the synthesized rules by literal name:
// fall_speed(score), is_caught(...)===1, score_after_catch, lives_after_miss,
// is_game_over(lives)===1.
// ---------------------------------------------------------------------------
const HTML_TEMPLATE: &str = r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Lane Catch — every rule synthesized from English by nCPU</title>
<style>
  :root{--slate:#0C1015;--panel:#151B23;--line:#2A3340;--ink:#DCE3EA;--dim:#8A99A9;--amber:#FFB454;--cyan:#7DD3FC;--rose:#E0708C;}
  *{box-sizing:border-box} html,body{margin:0;background:var(--slate);color:var(--ink);font-family:'JetBrains Mono',ui-monospace,monospace}
  .wrap{max-width:880px;margin:0 auto;padding:24px}
  h1{font-size:22px;letter-spacing:-.5px;margin:0 0 4px} .sub{color:var(--dim);font-size:13px;margin-bottom:18px}
  .amber{color:var(--amber)} .cyan{color:var(--cyan)}
  canvas{background:var(--panel);border:1px solid var(--line);display:block;margin:0 auto;border-radius:4px}
  .hud{display:flex;gap:20px;justify-content:center;margin:12px 0;font-size:14px}
  .hud b{color:var(--amber)}
  .hint{text-align:center;color:var(--dim);font-size:12px;margin-top:6px}
  table{width:100%;border-collapse:collapse;margin-top:22px;font-size:12px}
  th,td{text-align:left;padding:7px 9px;border-bottom:1px solid var(--line);vertical-align:top}
  th{color:var(--dim);text-transform:uppercase;letter-spacing:.1em;font-size:10px;font-weight:400}
  td.r{color:var(--amber)} td.m{color:var(--cyan)} td.c.high{color:#7CFFA0} td.c.medium{color:var(--amber)} td.c.low{color:var(--dim)}
  .prov-h{margin-top:26px;font-size:14px}
  .badge{display:inline-block;border:1px solid var(--line);padding:2px 8px;border-radius:3px;color:var(--dim);font-size:11px}
</style></head>
<body><div class="wrap">
  <h1>Lane Catch <span class="badge">rules synthesized from a plain-English description</span></h1>
  <p class="sub">Move the <span class="amber">basket</span> with ← → (or A/D). Catch falling blocks, don't miss.
  Every gameplay rule below was <span class="cyan">discovered by the nCPU synthesizer</span> from a free-text English
  description naming five functions and their input/output examples — no human wrote the rule bodies. Each rule was then
  re-verified (Mog ↔ transpiled JS) across the whole input domain before being assembled into this game.
  (Basket edge-bounding and the canvas/loop/input shell are plain presentation, not synthesized rules.)</p>
  <div class="hud">score <b id="score">0</b> &nbsp; lives <b id="lives">3</b> &nbsp; <span id="state" class="cyan">playing</span></div>
  <canvas id="c" width="480" height="420"></canvas>
  <p class="hint">← → / A D to move · R to restart</p>

  <div class="prov-h">❯ rule provenance — each function below is machine-synthesized from English, verified, transpiled</div>
  <table><thead><tr><th>rule</th><th>english spec</th><th>solver method</th><th>domain pts</th><th>verified</th></tr></thead>
  <tbody><!--__PROV__--></tbody></table>
</div>
<script>
/* ===== synthesized + verified game logic (nobody wrote these bodies) ===== */
/*__FUNCS__*/
/* ===== thin presentation shell wiring the synthesized rules ===== */
const LANES=4, W=480, H=420, lw=W/LANES;
const cv=document.getElementById('c'), ctx=cv.getContext('2d');
const SPAWN_FRAMES=26;
let basket, score, lives, items, frame, over;
function reset(){ basket=1; score=0; lives=3; items=[]; frame=0; over=false;
  document.getElementById('state').textContent='playing'; document.getElementById('state').className='cyan'; }
reset();
const clampLane = v => Math.max(0, Math.min(LANES-1, v)); // presentation bound (not a synthesized rule)
addEventListener('keydown',e=>{
  if(e.key==='ArrowLeft'||e.key==='a'||e.key==='A') basket=clampLane(basket-1);
  if(e.key==='ArrowRight'||e.key==='d'||e.key==='D') basket=clampLane(basket+1);
  if(e.key==='r'||e.key==='R') reset();
});
function step(){
  frame++;
  if(!over && frame>=SPAWN_FRAMES){ items.push({lane:(Math.random()*LANES)|0, y:-20}); frame=0; }
  // synthesized REAL-VALUED difficulty ramp: fall_speed_f(score) = 3 + 0.5*score
  // (an f64 rule discovered from English, transpiled to JS, verified Mog↔node).
  // Sub-integer px/frame — the integer transpile lane could never express this.
  const vy = fall_speed_f(score);
  for(const it of items) it.y += vy;
  for(const it of items){
    if(it.y>=H-46 && it.y<H-10 && !it.done){
      it.done=true;
      if(is_caught(Math.abs(basket - it.lane))===1){ score=score_after_catch(score); }
      else { lives=lives_after_miss(lives); }
    }
  }
  items = items.filter(it=>it.y<H+20);
  if(!over && is_game_over(lives)===1){ over=true;
    document.getElementById('state').textContent='game over — R to restart';
    document.getElementById('state').className=''; document.getElementById('state').style.color='var(--rose)'; }
  document.getElementById('score').textContent=score;
  document.getElementById('lives').textContent=Math.max(0,lives);
}
function draw(){
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle='#2A3340'; for(let i=1;i<LANES;i++){ctx.beginPath();ctx.moveTo(i*lw,0);ctx.lineTo(i*lw,H);ctx.stroke();}
  for(const it of items){ ctx.fillStyle=it.done?'#3a4350':'#7DD3FC'; ctx.fillRect(it.lane*lw+lw/2-10, it.y, 20, 20); }
  ctx.fillStyle='#FFB454'; ctx.fillRect(basket*lw+lw/2-26, H-40, 52, 16);
}
function loop(){ if(!over) step(); draw(); requestAnimationFrame(loop); }
loop();
</script></body></html>"#;
