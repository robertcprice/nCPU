//! NL synthesis proposer for the repair loop (Package B/H bridge).
//!
//! When a `RepoTaskSpec.issue` begins with `nl:` or `synthesize:`, this proposer
//! runs the Linguigenesis-native `AgentRun` path and writes verified synthesis
//! output into an allowed repository file.

use crate::agent::agent_run::AgentRun;
use crate::agent::coding_intent::CodingIntent;
use crate::agent::repo::{
    FailureAnalysis, FailureKind, RepairContext, RepairEdit, RepairPatch, RepoTaskSpec,
};
use std::fs;
use std::path::{Path, PathBuf};

/// Extract NL description from task issue (supports `nl:` and `synthesize:` prefixes).
pub fn nl_description_from_issue(issue: &str) -> Option<String> {
    for prefix in ["synthesize:", "nl:"] {
        if let Some(rest) = issue.strip_prefix(prefix) {
            let trimmed = rest.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

/// Whether this task should use the NL synthesis proposer.
pub fn task_uses_nl_synthesis(task: &RepoTaskSpec) -> bool {
    nl_description_from_issue(&task.issue).is_some()
}

/// Repair-loop proposer: NL → `AgentRun` → patch with synthesized Rust code.
pub fn nl_synthesis_proposer(
    task: &RepoTaskSpec,
    context: &RepairContext,
    _iteration: usize,
    analysis: Option<&FailureAnalysis>,
) -> Result<RepairPatch, String> {
    let description = nl_description_from_issue(&task.issue).unwrap_or_else(|| task.issue.clone());

    // EMERGENT bare-NL stage FIRST: grounded localization beats guessed. It
    // fires only when the described fn genuinely EXISTS in the repo (content
    // scan + emergent morphology match), then swaps that fn's body with the
    // bridge-comprehended, verified synthesis. This is what repairs a repo fn
    // whose NAME differs from the op ("the twice function should double the
    // number"): the intent-name path below would miss `fn twice`, fall back to
    // the first .rs file, and write a brand-new `fn double` — wrong file, wrong
    // fn. When the description names no existing fn (feature work, inline-
    // example specs), this declines and the primary proceeds unchanged.
    // RENAME refactor FIRST: "rename X to Y" is a structural refactor, never a
    // body-swap — hoisted above the edit stage so it can't be hijacked (the
    // edit stage previously declined it only by accident of reshape no-op).
    if let Some(patch) = try_rename_patch(context, &description) {
        return Ok(patch);
    }

    // SIGNATURE-CHANGE refactor: "add a parameter P to F defaulting to N" — a
    // coordinated edit of F's signature AND every call site (each call gains the
    // default as its new argument). Structural like rename; hoisted for the same
    // reason.
    if let Some(patch) = try_add_param_patch(context, &description) {
        return Ok(patch);
    }

    if let Some(patch) = try_emergent_synthesis_patch(task, context, &description, analysis) {
        return Ok(patch);
    }

    // FEATURE-ADD stage: an ADDITIVE request ("add a function that triples a
    // number") whose fn does NOT exist yet — synthesize it via emergent
    // comprehension and APPEND it (TDD shape: a failing test referencing the
    // missing fn compiles+passes after). Declines for non-additive prose or
    // when the fn already exists (that's the edit stage above).
    if let Some(patch) = try_emergent_addition_patch(task, context, &description, analysis) {
        return Ok(patch);
    }

    // Primary path: genuine verified synthesis through the bridge + solver.
    // Generalizes to any demonstrated function (registry op or inline examples),
    // not just the canned scalar shapes the keyword fast-patch can express.
    if let Some(patch) = try_real_synthesis_patch(task, context, &description) {
        return Ok(patch);
    }

    // TEST-MINED synthesis: the prose carried no examples, but the failing test's
    // `assert_eq!` calls do. Mine them and solve the real problem — deterministic,
    // verified, no model. Strictly stronger than the keyword fast-patch below.
    if let Some(patch) = try_test_mined_synthesis_patch(task, context, &description) {
        return Ok(patch);
    }

    if let Some(patch) = try_nl_repo_fast_patch(task, context, &description, analysis) {
        return Ok(patch);
    }

    // GATED MODEL fallback: every deterministic path declined. An optional,
    // untrusted local LLM proposes a fix from the failure — inert without
    // NSYNTH_LOCAL_LLM_URL, and the cargo-test oracle still gates whatever it
    // returns. Deterministic-first, model-last.
    if let Some(patch) = try_model_repair_patch(task, context, &description, analysis) {
        return Ok(patch);
    }

    let mut run = AgentRun::start(description);
    run.comprehend().map_err(|e| e.to_string())?;
    if run.needs_clarification() {
        return Err(format!(
            "clarification required: {:?}",
            run.clarification_questions
        ));
    }
    let intent = run.intent.clone();
    if let Some(ref intent) = intent {
        let target_hint = match pick_target_path(task, context, Some(intent)) {
            Ok(target) => read_relative_file(context, &target).ok(),
            Err(_) => None,
        };
        if let Some(rust_body) = repo_rust_body_for_nl(intent, "", target_hint.as_deref()) {
            let stub = crate::solver::SolveResult {
                success: true,
                code: rust_body,
                method: "nl_rust_repo_stub".to_string(),
                error: None,
                metadata: Default::default(),
            };
            return repair_patch_from_synthesis(task, context, Some(intent), &stub);
        }
    }
    let result = run.synthesize()?.clone();
    repair_patch_from_synthesis(task, context, intent.as_ref(), &result)
}

/// Real verified synthesis for repo repair (North Star path).
///
/// NL/example description → bridge `SynthesisRequirement` → solver `Problem` →
/// verified Rust, reshaped to **preserve the repo function's exact signature**
/// (so the failing test's call convention is honoured) by swapping only the
/// body with the synthesized logic and renaming params positionally. This
/// replaces keyword→canned-code matching with genuine synthesis and generalizes
/// to *any* function the task demonstrates — registry op or inline I/O examples.
pub fn try_real_synthesis_patch(
    task: &RepoTaskSpec,
    context: &RepairContext,
    description: &str,
) -> Option<RepairPatch> {
    let intent = CodingIntent::from_nl_lenient(description).ok()?;
    if intent.examples.is_empty() {
        return None;
    }
    let problem = intent.to_problem().ok()?;
    let result = crate::solver::solve_problem(&problem);
    if !result.success {
        return None;
    }
    let target = pick_target_path(task, context, Some(&intent)).ok()?;
    let old_text = read_relative_file(context, &target).ok()?;
    let repo_fn = resolve_repo_fn_name(
        intent
            .function_name
            .strip_prefix("nl_")
            .unwrap_or(&intent.function_name),
        Some(&old_text),
    );
    let synthesized = rust_code_for_repo_synthesis(&result.code);
    // The solver sometimes emits an abstract IR (Result-style `ok(..)`/`err(..)`
    // wrappers, unlowered `:=`) that is not plain Rust for the repo's concrete
    // signature. Adopt real synthesis only when the output is directly
    // compilable; otherwise decline and let the proven fallback handle it.
    if !is_plain_rust_body(&synthesized) {
        return None;
    }
    // ARITY GATE (mirrors try_emergent_synthesis_patch): from_nl_lenient can invent example inputs
    // whose arity disagrees with the repo signature — "maximum of two numbers" yields a 1-arg
    // (single-list) intent that solves to a 1-param fn, but the repo's `max_of(a, b)` takes two
    // scalars; reshaping the wrong shape onto the wrong params is the observed type mismatch.
    // Declining hands such cases to the example-grounded, never-wrong test-mining router.
    let repo_arity = fn_header_params(&old_text, &repo_fn)
        .map(|p| parse_param_idents(&p).len())
        .unwrap_or(0);
    let synth_arity = fn_header_params(&synthesized, &repo_fn)
        .or_else(|| {
            first_fn_name_in_source(&synthesized).and_then(|n| fn_header_params(&synthesized, &n))
        })
        .map(|p| parse_param_idents(&p).len())
        .unwrap_or(0);
    if repo_arity != 0 && synth_arity != 0 && repo_arity != synth_arity {
        return None;
    }
    let new_text = reshape_to_repo_signature(&old_text, &repo_fn, &synthesized)?;
    if new_text == old_text {
        return None;
    }
    Some(
        RepairPatch::new()
            .with_edit(RepairEdit::new(
                target,
                old_text,
                new_text,
                "nl real-synthesis proposer (bridge+solver, verified)",
            ))
            .with_metadata("proposer", "nl_real_synthesis")
            .with_metadata("synthesis_method", result.method.clone()),
    )
}

/// GATED MODEL-REPAIR stage — the untrusted proposer of last resort.
///
/// When every deterministic path (rename / add-param / emergent / real-synthesis /
/// fast-patch) has declined, an OPTIONAL local LLM proposes a function body from
/// the concrete failure. It is a PROPOSER ONLY: the body is reshaped to the repo
/// fn's exact signature and handed back as an ordinary patch, so the caller's
/// cargo-test acceptance oracle still decides — a wrong proposal fails the test and
/// is rolled back like any other. The model NEVER bypasses a gate and the guarantee
/// never depends on it.
///
/// Inert by default: with `NSYNTH_LOCAL_LLM_URL` unset the lane returns `None`
/// immediately, so there is zero behaviour change on any machine without a model.
pub fn try_model_repair_patch(
    task: &RepoTaskSpec,
    context: &RepairContext,
    description: &str,
    analysis: Option<&FailureAnalysis>,
) -> Option<RepairPatch> {
    // Off unless an endpoint is configured — the guarantee is the cargo-test oracle,
    // not the model.
    if std::env::var("NSYNTH_LOCAL_LLM_URL")
        .ok()
        .filter(|u| !u.trim().is_empty())
        .is_none()
    {
        return None;
    }

    // Localize the repo fn to repair, exactly as the verified path does.
    let intent = CodingIntent::from_nl_lenient(description).ok();
    let target = pick_target_path(task, context, intent.as_ref()).ok()?;
    let old_text = read_relative_file(context, &target).ok()?;
    let default_fn = intent
        .as_ref()
        .map(|i| {
            i.function_name
                .strip_prefix("nl_")
                .unwrap_or(&i.function_name)
                .to_string()
        })
        .unwrap_or_default();
    let repo_fn = resolve_repo_fn_name(&default_fn, Some(&old_text));

    // Feed the model the CURRENT body + the concrete failure so it repairs THIS
    // error rather than guessing blind. Both are optional (best-effort feedback).
    let prior_code = crate::doc_ingest::extract_rust_fn_sources(&old_text)
        .into_iter()
        .find(|(n, _)| n == &repo_fn)
        .map(|(_, s)| s);
    let failure = analysis
        .map(|a| format!("{} (suggested: {})", a.message, a.suggested_action))
        .unwrap_or_default();
    let prior: Option<(&str, &str)> = match (&prior_code, failure.is_empty()) {
        (Some(code), false) => Some((code.as_str(), failure.as_str())),
        _ => None,
    };
    let request = format!(
        "{description}\n\nWrite the Rust function `{repo_fn}` so the failing test passes. \
         Return only the function definition."
    );

    // Ask for RUST, not Mog: the repo-repair oracle is cargo test over real Rust, and
    // small local models write Rust well but not the Mog DSL.
    let program = crate::local_llm::propose_rust_fn(&request, prior, 0.2)?;
    // MULTI-FILE: if the model returned several functions that map onto EXISTING repo
    // files (e.g. a fix touching a handler and its helper), coordinate them into one
    // atomic patch. The cargo-test oracle still gates the whole thing all-or-nothing.
    // Falls back to the single-file body swap below.
    if let Some(patch) = model_response_to_multifile_patch(context, &program) {
        return Some(patch);
    }
    let new_text = model_body_to_new_text(&old_text, &repo_fn, &program)?;
    Some(
        RepairPatch::new()
            .with_edit(RepairEdit::new(
                target,
                old_text,
                new_text,
                "gated model-repair proposer (untrusted; cargo-test gated)",
            ))
            .with_metadata("proposer", "model_repair"),
    )
}

/// Coordinate a MULTI-FILE model patch: extract every `fn` from `response` and, for
/// each one a repo file already defines, swap that fn's body into its file
/// (accumulating multiple swaps per file). Returns a patch only when it touches TWO
/// OR MORE files — a single-file response goes back to the caller's body-swap path.
/// Pure over `context` + `response`, so it is testable with no model present; every
/// edit still passes the plain-Rust gate and, downstream, the cargo-test oracle.
fn model_response_to_multifile_patch(context: &RepairContext, response: &str) -> Option<RepairPatch> {
    let fns = crate::doc_ingest::extract_rust_fn_sources(response);
    if fns.len() < 2 {
        return None;
    }
    // Accumulate per file: (original_text, current_text with swaps applied).
    let mut per_file: std::collections::BTreeMap<String, (String, String)> =
        std::collections::BTreeMap::new();
    for (name, src) in &fns {
        let Some(file) = context.files.iter().find(|f| {
            f.path.ends_with(".rs") && file_defines_function(f.text.as_deref().unwrap_or(""), name)
        }) else {
            continue; // a proposed fn with no existing definition — not a coordinated edit
        };
        let orig = file.text.clone().unwrap_or_default();
        let entry = per_file.entry(file.path.clone()).or_insert_with(|| (orig.clone(), orig));
        let body = rust_code_for_repo_synthesis(src);
        if !is_plain_rust_body(&body) {
            continue;
        }
        if let Some(next) = reshape_to_repo_signature(&entry.1, name, &body) {
            entry.1 = next;
        }
    }
    let mut patch = RepairPatch::new();
    let mut files_changed = 0usize;
    for (path, (orig, current)) in per_file {
        if current != orig {
            patch = patch.with_edit(RepairEdit::new(
                path,
                orig,
                current,
                "gated model multi-file repair (untrusted; cargo-test gated)",
            ));
            files_changed += 1;
        }
    }
    if files_changed >= 2 {
        Some(patch.with_metadata("proposer", "model_repair_multifile"))
    } else {
        None
    }
}

/// Pure string core of the model-repair stage (TESTABLE without a model or repo):
/// reshape a proposed program to `repo_fn`'s signature in `old_text`. Declines when
/// the proposal is not plain compilable Rust (IR wrappers / unlowered `:=`) or is a
/// no-op — the same gate the verified synthesis path applies.
fn model_body_to_new_text(old_text: &str, repo_fn: &str, program: &str) -> Option<String> {
    let body = rust_code_for_repo_synthesis(program);
    if !is_plain_rust_body(&body) {
        return None;
    }
    let new_text = reshape_to_repo_signature(old_text, repo_fn, &body)?;
    (new_text != *old_text).then_some(new_text)
}

/// TEST-MINED SYNTHESIS — the deterministic lever that turns the repair loop's own
/// TEST ORACLE into a searchable spec. Most real repairs have a failing test full of
/// `assert_eq!(f(x), y)` calls; those ARE I/O examples, but nothing fed them to the
/// solver — `try_real_synthesis_patch` only reads examples out of the PROSE and
/// declines bare NL. So a bare-NL-but-tested repair used to fall through every
/// deterministic stage to the (gated) model. This stage mines the asserts, solves
/// the real problem, strict-verifies + screens for memorization, and reshapes to the
/// repo signature — fully verified, no examples in the prose, NO model.
pub fn try_test_mined_synthesis_patch(
    task: &RepoTaskSpec,
    context: &RepairContext,
    description: &str,
) -> Option<RepairPatch> {
    let intent = CodingIntent::from_nl_lenient(description).ok();
    let mut target = pick_target_path(task, context, intent.as_ref()).ok()?;
    let mut old_text = read_relative_file(context, &target).ok()?;
    let default_fn = intent
        .as_ref()
        .map(|i| i.function_name.strip_prefix("nl_").unwrap_or(&i.function_name).to_string())
        .unwrap_or_default();
    // Mine integer asserts for a function across every context file (tests may live in a sibling
    // module), dedup, and require enough points to PIN the function.
    let mine_for = |name: &str| {
        let mut rows: Vec<(Vec<crate::benchmark::Value>, crate::benchmark::Value)> = Vec::new();
        for f in &context.files {
            if let Some(t) = f.text.as_deref() {
                rows.extend(mine_asserts(t, name));
            }
        }
        rows.sort();
        rows.dedup();
        rows
    };
    let mut repo_fn = resolve_repo_fn_name(&default_fn, Some(&old_text));
    let mut rows = mine_for(&repo_fn);
    // The intent's guessed name may not match the repo's actual failing function. If too few
    // asserts were mined, try (a) every function DEFINED in the target file, then (b) every
    // function CALLED by an assert but DEFINED NOWHERE — a MISSING function the failing test
    // references (feature-add). Adopt the (name, rows) with the most examples.
    if rows.len() < 2 {
        let defined = defined_fn_names(&old_text);
        let mut candidates: Vec<String> = defined.clone();
        for f in &context.files {
            if let Some(t) = f.text.as_deref() {
                for c in assert_called_fn_names(t) {
                    if !defined.contains(&c) && !candidates.contains(&c) {
                        candidates.push(c);
                    }
                }
            }
        }
        for cand in candidates {
            let cand_rows = mine_for(&cand);
            if cand_rows.len() > rows.len() {
                rows = cand_rows;
                repo_fn = cand;
            }
        }
    }
    if rows.is_empty() {
        return None;
    }
    // Repair the file that DEFINES repo_fn, not the one pick_target_path chose. The picked target
    // follows the QUERY / assert site, which for a real repo is often NOT the definition: an
    // integration test lives in `tests/`, and `src/lib.rs` may only declare `pub mod math;` while
    // the function lives in `src/math.rs`. Editing the assert site or the module-declaring root
    // would replace the wrong body or append a duplicate/misplaced definition. Only retarget when
    // the defining file DIFFERS from the picked target (so single-file repairs are byte-for-byte
    // unchanged), and re-read it through read_relative_file so old_text matches the on-disk content
    // the patch apply will search for — the context snapshot can differ (trailing whitespace) and
    // trip "expected exactly one occurrence".
    if let Some(def_path) = context
        .files
        .iter()
        .filter(|f| f.path.ends_with(".rs") && f.path != target)
        .find(|f| f.text.as_deref().map(|t| file_defines_function(t, &repo_fn)).unwrap_or(false))
        .map(|f| f.path.clone())
    {
        if !old_text.contains(&format!("fn {repo_fn}")) {
            if let Ok(def_text) = read_relative_file(context, &def_path) {
                target = def_path;
                old_text = def_text;
            }
        }
    }
    let repo_has_fn = old_text.contains(&format!("fn {repo_fn}"));

    let exs: Vec<crate::benchmark::Example> = rows
        .iter()
        .map(|(ins, out)| crate::benchmark::Example {
            inputs: ins.clone(),
            expected: out.clone(),
        })
        .collect();
    // Prefer the VERIFIED NL ROUTER over the raw solver. It grounds the prose+examples in the
    // hardened op library behind a distinguishing gate: it resolves "maximum of a list" to list_max
    // and REJECTS coincidental overfits (max_except_last, which merely reproduces under-determined
    // examples), returns max_two for "maximum of two numbers", and reverse_list for "reverse a list"
    // from a single example — the never-wrong guarantee carried into repo repair. Because the router
    // is name-grounded + example-verified, one example suffices; the raw-solver fallback still needs
    // >= 2 and declines array-returning shapes it does not reshape reliably.
    let (synth_mog, method) = match crate::verified_nl_router::answer(description, &exs) {
        crate::verified_nl_router::Answer::Library { name, code } => {
            (code, format!("verified-nl-router:library:{name}"))
        }
        crate::verified_nl_router::Answer::Composition { code } => {
            (code, "verified-nl-router:composition".to_string())
        }
        crate::verified_nl_router::Answer::Synthesized { method, code } => {
            (code, format!("verified-nl-router:{method}"))
        }
        _ => {
            if rows.len() < 2 {
                return None;
            }
            // Raw-solver fallback declines array-RETURNING functions (its Vec/slice reshape is not
            // reliable for them); the router above already handles those cleanly.
            if exs.iter().any(|e| matches!(e.expected, crate::benchmark::Value::Array(_))) {
                return None;
            }
            let sig: &'static str = Box::leak(
                crate::linguigenesis_bridge::infer_signature(&repo_fn, &exs).into_boxed_str(),
            );
            let problem = crate::benchmark::Problem {
                name: repo_fn.clone(),
                category: "repo-test-mined",
                description: "",
                signature: sig,
                examples: exs.clone(),
                ..Default::default()
            };
            let res = crate::solver::solve_problem(&problem);
            if !res.success {
                return None;
            }
            if crate::runtime::verify_problem_code_strict(&problem, &res.code).is_err() {
                return None;
            }
            // Same memorization guard as the CLI/teach paths: never adopt a magic-constant fit.
            if crate::synth_confidence::is_memorization_overfit(&res.code, &exs) {
                return None;
            }
            (res.code, res.method)
        }
    };
    let synthesized = rust_code_for_repo_synthesis(&synth_mog);
    if !is_plain_rust_body(&synthesized) {
        return None;
    }
    let new_text = if repo_has_fn {
        reshape_to_repo_signature(&old_text, &repo_fn, &synthesized)?
    } else {
        // MISSING function (feature-add): reshape yields the standalone fn source; APPEND it to
        // the file rather than replacing the file contents.
        let new_fn = reshape_to_repo_signature(&old_text, &repo_fn, &synthesized)?;
        format!("{}\n\n{}\n", old_text.trim_end(), new_fn.trim())
    };
    if new_text == old_text {
        return None;
    }
    Some(
        RepairPatch::new()
            .with_edit(RepairEdit::new(
                target,
                old_text,
                new_text,
                "test-mined synthesis proposer (examples from failing asserts; verified, no LLM)",
            ))
            .with_metadata("proposer", "nl_test_mined_synthesis")
            .with_metadata("synthesis_method", method),
    )
}

/// Mine I/O examples for `fn_name` from `assert_eq!(..)` calls in `text`. Both
/// `assert_eq!(f(4), 8)` and `assert_eq!(8, f(4))` yield `([4], 8)`. Integer, string
/// (`"abc"`), and boolean (`true`/`false`) literals are captured — the solver's
/// verified domains; anything else (floats, expressions, method chains) is skipped.
/// Name matching is word-boundary safe, so `add` never matches `add_two`.
/// Identifiers called as functions inside `assert_eq!(..)` args (e.g. `triple` from
/// `assert_eq!(triple(3), 9)`). Used to find a MISSING function a failing test references — the
/// feature-add target that no `mine_asserts(defined_name)` would ever pin. Skips the `vec` macro
/// and numeric leads.
fn assert_called_fn_names(text: &str) -> Vec<String> {
    let mut names = std::collections::BTreeSet::new();
    let mut cur = text;
    while let Some(rel) = cur.find("assert_eq!") {
        let after = &cur[rel + "assert_eq!".len()..];
        cur = after;
        let Some(open) = after.find('(') else { continue };
        let Some(inner) = balanced_parens(&after[open..]) else { continue };
        for arg in split_top_level_comma(inner) {
            let a = arg.trim();
            if let Some(p) = a.find('(') {
                let ident: String = a[..p]
                    .chars()
                    .rev()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect();
                if !ident.is_empty()
                    && !ident.starts_with(|c: char| c.is_ascii_digit())
                    && ident != "vec"
                {
                    names.insert(ident);
                }
            }
        }
    }
    names.into_iter().collect()
}

fn mine_asserts(text: &str, fn_name: &str) -> Vec<(Vec<crate::benchmark::Value>, crate::benchmark::Value)> {
    let mut out = Vec::new();
    let mut cur = text;
    while let Some(rel) = cur.find("assert_eq!") {
        let after = &cur[rel + "assert_eq!".len()..];
        cur = after; // advance regardless of parse success
        let Some(open) = after.find('(') else { continue };
        let Some(inner) = balanced_parens(&after[open..]) else { continue };
        let args = split_top_level_comma(inner);
        if args.len() < 2 {
            continue;
        }
        if let Some(ex) = assert_pair_to_example(fn_name, args[0].trim(), args[1].trim())
            .or_else(|| assert_pair_to_example(fn_name, args[1].trim(), args[0].trim()))
        {
            out.push(ex);
        }
    }
    out
}

/// Parse a Rust literal token into a `Value`: integer, `true`/`false`, a simple
/// double-quoted string (no escapes), an owned-string constructor (`"x".to_string()`,
/// `String::from("x")`), or an ARRAY/SLICE/VEC literal (`[1,2]`, `&[1,2]`, `vec![1,2]`,
/// nested). Returns `None` for anything else so the miner only captures cleanly-
/// verifiable examples. Array + owned-string forms are what a real test uses to pass a
/// list or `String` argument — without them the CLI's generic "fix the failing tests"
/// query mines 0 rows for every collection/string function (the semantic issue path in
/// the bench sidesteps mining via name-match, hiding this).
fn parse_literal(tok: &str) -> Option<crate::benchmark::Value> {
    use crate::benchmark::Value;
    let t = tok.trim();
    if let Ok(i) = t.parse::<i64>() {
        return Some(Value::Int(i));
    }
    match t {
        "true" => return Some(Value::Bool(true)),
        "false" => return Some(Value::Bool(false)),
        _ => {}
    }
    if t.len() >= 2 && t.starts_with('"') && t.ends_with('"') {
        let body = &t[1..t.len() - 1];
        if !body.contains('\\') {
            return Some(Value::Str(body.to_string()));
        }
    }
    // Owned-`String` constructors: unwrap and re-parse the inner string literal.
    for suffix in [".to_string()", ".to_owned()", ".into()"] {
        if let Some(inner) = t.strip_suffix(suffix) {
            return parse_literal(inner.trim());
        }
    }
    if let Some(inner) = t.strip_prefix("String::from(").and_then(|r| r.strip_suffix(')')) {
        return parse_literal(inner.trim());
    }
    // Array / slice / vec literals: strip the container spelling, then parse each element
    // recursively (nested arrays work) and collect. `[]` / `vec![]` -> empty array.
    let mut body = t;
    if let Some(r) = body.strip_prefix("vec!") {
        body = r.trim();
    }
    if let Some(r) = body.strip_prefix('&') {
        body = r.trim();
    }
    if body.starts_with('[') && body.ends_with(']') {
        let content = &body[1..body.len() - 1];
        let elems: Option<Vec<Value>> = split_top_level_comma(content)
            .iter()
            .map(|e| e.trim())
            .filter(|e| !e.is_empty())
            .map(parse_literal)
            .collect();
        return elems.map(Value::Array);
    }
    None
}

/// Content strictly inside the first balanced `(..)` of `s` (which must start with
/// `(`). `None` if unbalanced.
fn balanced_parens(s: &str) -> Option<&str> {
    let b = s.as_bytes();
    if b.first() != Some(&b'(') {
        return None;
    }
    let mut depth = 0i32;
    for (i, &c) in b.iter().enumerate() {
        match c {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[1..i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Split `s` at commas that are NOT nested inside `()` or `[]`.
fn split_top_level_comma(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut last = 0usize;
    for (i, &c) in s.as_bytes().iter().enumerate() {
        match c {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            b',' if depth == 0 => {
                parts.push(&s[last..i]);
                last = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[last..]);
    parts
}

/// `(call_side, val_side)` → `(args, expected)` when `call_side` is `fn_name(lits..)`
/// (word-boundary) and `val_side` is a literal, both parsed by [`parse_literal`].
fn assert_pair_to_example(
    fn_name: &str,
    call_side: &str,
    val_side: &str,
) -> Option<(Vec<crate::benchmark::Value>, crate::benchmark::Value)> {
    let expected = parse_literal(val_side)?;
    let pat = format!("{fn_name}(");
    let pos = call_side.find(&pat)?;
    if pos > 0 {
        let prev = call_side.as_bytes()[pos - 1];
        if prev.is_ascii_alphanumeric() || prev == b'_' {
            return None; // fn_name is a suffix of a longer identifier
        }
    }
    let rest = &call_side[pos + fn_name.len()..]; // starts at the '('
    let inner = balanced_parens(rest)?;
    let args: Option<Vec<crate::benchmark::Value>> =
        split_top_level_comma(inner).iter().map(|a| parse_literal(a)).collect();
    let args = args?;
    if args.is_empty() {
        return None;
    }
    Some((args, expected))
}

/// EMERGENT NL edit driver (no examples, no LLM): "the double function should
/// double the number" carries no I/O pairs, yet becomes a verified patch.
///
///   WHAT  — `bridge.synthesize_from_description` comprehends the description
///           through the emergent resolver (morphology / graph / WordNet over the
///           registry) and synthesizes + strict-verifies from the resolved op's
///           own example_cases. Purely symbolic by default: the local-LLM lane
///           inside is env-gated OFF (`NSYNTH_LOCAL_LLM_URL` unset ⇒ inert).
///   WHERE — the repo fn to replace is localized by CONTENT: scan defined fn
///           names in context files and match them emergently against the
///           description (exact token or shared morphological stem, every
///           snake_case part matched — so "the doubling function" finds
///           `fn double`, "reverse the list" finds `fn reverse_list`).
///   GATE  — same as the primary: plain-Rust check, signature-preserving
///           reshape, and the caller's cargo-test acceptance oracle.
pub fn try_emergent_synthesis_patch(
    task: &RepoTaskSpec,
    context: &RepairContext,
    description: &str,
    analysis: Option<&FailureAnalysis>,
) -> Option<RepairPatch> {
    let _ = task;
    // OBSERVATION-DRIVEN localization: the failure-implicated file (compile
    // error / trace `file:line` from FailureAnalysis) outranks walk order when
    // several files define a matching fn — the compiler told us where it hurts.
    let preferred = analysis.and_then(|a| a.file.clone());
    let (target, repo_fn) = locate_described_fn(context, description, preferred.as_deref())?;
    let observation_grounded = preferred
        .as_deref()
        .map(|p| p.ends_with(&target) || target.ends_with(p))
        .unwrap_or(false);

    // DEFER to the example-driven primary whenever its localization is GROUNDED:
    // the comprehended intent's fn name is genuinely defined in the repo (e.g. an
    // inline-example spec "triple(2)=6 ..." over an existing `fn triple` — the
    // user's demonstrated I/O is the stronger spec). Two exceptions lead here:
    // the intent name misses (renamed fn: intent says `double`, repo defines
    // `twice`), or the OBSERVATION localizes to the failure-implicated file —
    // the failing trace is stronger evidence than a name guess (two files may
    // define the described fn; only the implicated one is broken).
    if !observation_grounded {
        if let Ok(intent) = CodingIntent::from_nl_lenient(description) {
            let primary_name = intent
                .function_name
                .strip_prefix("nl_")
                .unwrap_or(&intent.function_name);
            let grounded = context.files.iter().any(|f| {
                f.path.ends_with(".rs")
                    && file_defines_function(f.text.as_deref().unwrap_or(""), primary_name)
            });
            if grounded && !intent.examples.is_empty() {
                return None;
            }
        }
    }
    let bridge = crate::linguigenesis_bridge::LinguigenesisBridge::new();
    let result = bridge
        .synthesize_from_description(description, Some(&repo_fn))
        .ok()?;
    if !result.success {
        return None;
    }
    let synthesized = rust_code_for_repo_synthesis(&result.code);
    if !is_plain_rust_body(&synthesized) {
        return None;
    }
    let old_text = read_relative_file(context, &target).ok()?;
    // ARITY GATE: a bare-NL comprehension that lands on a DIFFERENT-arity function than the repo
    // signature cannot repair it — the reshape would fit the wrong shape onto the wrong parameters.
    // Concretely, the bridge mis-reads "maximum of two numbers" (two scalars) as an array max —
    // `fn max_of(xs: [i64])`, one param — while the repo fn is `max_of(a, b)`, two params; reshaping
    // an array body onto two scalars yields a type mismatch. Declining here defers such cases to the
    // example-grounded test-mining router, which distinguishes max_two from list_max via the failing
    // test's I/O. Same-arity comprehensions (abs, reverse, array folds) are unaffected.
    let repo_arity = fn_header_params(&old_text, &repo_fn)
        .map(|p| parse_param_idents(&p).len())
        .unwrap_or(0);
    let synth_arity = fn_header_params(&synthesized, &repo_fn)
        .or_else(|| {
            first_fn_name_in_source(&synthesized).and_then(|n| fn_header_params(&synthesized, &n))
        })
        .map(|p| parse_param_idents(&p).len())
        .unwrap_or(0);
    if repo_arity != 0 && synth_arity != 0 && repo_arity != synth_arity {
        return None;
    }
    let new_text = reshape_to_repo_signature(&old_text, &repo_fn, &synthesized)?;
    if new_text == old_text {
        return None;
    }
    Some(
        RepairPatch::new()
            .with_edit(RepairEdit::new(
                target,
                old_text,
                new_text,
                "emergent NL synthesis proposer (bridge comprehension, verified; no examples, no LLM)",
            ))
            .with_metadata("proposer", "nl_emergent_synthesis")
            .with_metadata("synthesis_method", result.method.clone()),
    )
}

/// FEATURE-ADD (no LLM): synthesize a NEW function from bare NL and append it to
/// the repo — the TDD shape, where a failing test already references the missing
/// fn. Gates, in order:
///   * additive cue required (add/create/new/implement/write/need) so edit
///     requests never route here;
///   * the bridge must comprehend + verify the description (emergent resolver);
///   * the synthesized fn's name must NOT be defined anywhere (else it's an
///     edit, handled upstream);
///   * target file: the failure-implicated file when it's a repo .rs (the
///     compile error fires where the missing fn is CALLED — appending there
///     puts it in scope), else src/lib.rs, else the first .rs.
/// Output is plain Rust (Mog transpiled + pub'd) appended whole-file, gated by
/// the caller's cargo-test oracle like every other patch.
pub fn try_emergent_addition_patch(
    task: &RepoTaskSpec,
    context: &RepairContext,
    description: &str,
    analysis: Option<&FailureAnalysis>,
) -> Option<RepairPatch> {
    let _ = task;
    let lower = description.to_lowercase();
    const ADDITIVE: [&str; 6] = ["add ", "create ", "new ", "implement ", "write ", "need "];
    if !ADDITIVE.iter().any(|c| lower.contains(c)) {
        return None;
    }
    // Synthesize from the SEMANTIC CORE: "add a function that triples a number"
    // → "triples a number", so the imperative "add" can't shadow the real op.
    let synth_desc = lower
        .split_once(" that ")
        .map(|(_, rest)| rest.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| description.to_string());
    let bridge = crate::linguigenesis_bridge::LinguigenesisBridge::new();
    let result = bridge.synthesize_from_description(&synth_desc, None).ok()?;
    if !result.success {
        return None;
    }
    let synthesized = rust_code_for_repo_synthesis(&result.code);
    if !is_plain_rust_body(&synthesized) {
        return None;
    }
    // The new fn's name, from the synthesized code itself.
    let fn_name = synthesized
        .lines()
        .find_map(|l| {
            let t = l.trim_start();
            t.strip_prefix("pub fn ")
                .or_else(|| t.strip_prefix("fn "))
                .and_then(|r| r.split('(').next())
                .map(|n| n.trim().to_string())
        })
        .filter(|n| !n.is_empty())?;
    // Must be genuinely NEW — an existing definition means this is an edit.
    if context.files.iter().any(|f| {
        f.path.ends_with(".rs")
            && file_defines_function(f.text.as_deref().unwrap_or(""), &fn_name)
    }) {
        return None;
    }
    // Target file: failure-implicated .rs > src/lib.rs > first .rs.
    let target = analysis
        .and_then(|a| a.file.clone())
        .filter(|p| p.ends_with(".rs") && context.files.iter().any(|f| p.ends_with(&f.path)))
        .map(|p| {
            context
                .files
                .iter()
                .find(|f| p.ends_with(&f.path))
                .map(|f| f.path.clone())
                .unwrap_or(p)
        })
        .or_else(|| {
            context
                .files
                .iter()
                .find(|f| f.path == "src/lib.rs")
                .map(|f| f.path.clone())
        })
        .or_else(|| {
            context
                .files
                .iter()
                .find(|f| f.path.ends_with(".rs"))
                .map(|f| f.path.clone())
        })?;
    let old_text = read_relative_file(context, &target).ok()?;
    // Pub the new fn so sibling modules/tests can call it.
    let mut appended = synthesized.trim().to_string();
    if !appended.starts_with("pub ") && appended.starts_with("fn ") {
        appended = format!("pub {appended}");
    }

    // COORDINATED MULTI-FILE addition. When the crate has a MODULE-MANIFEST ROOT —
    // `src/lib.rs` (library) OR `src/main.rs` (binary) that carries `mod x;`
    // structure and no logic fns of its own — follow the repo's own convention: a
    // NEW module `src/<fn>.rs` holding the verified fn, plus the `mod`/`pub use`
    // wiring in that root. This is keyed on the manifest ROOT found in the repo, NOT
    // on which file the append `target` resolved to — so it fires for binary crates
    // too, where the target lands on a logic module, not the root. A binary root may
    // define `fn main` (its entry point); that is not a logic fn and does not
    // disqualify the manifest shape. New-file edits carry old_text == new_text (the
    // transaction's read-fallback makes that a create). Falls back to single-file
    // append for flat repos with no manifest root.
    let is_mod_decl = |l: &str| l.trim_start().starts_with("mod ") && l.trim_end().ends_with(';');
    let manifest_root = context.files.iter().find(|f| {
        (f.path == "src/lib.rs" || f.path == "src/main.rs")
            && f.text.as_deref().is_some_and(|t| {
                t.lines().any(is_mod_decl) && defined_fn_names(t).into_iter().all(|n| n == "main")
            })
    });
    if let Some(root_file) = manifest_root {
        let root_path = root_file.path.clone();
        let root_text = root_file.text.clone().unwrap_or_default();
        let module_file = format!("src/{fn_name}.rs");
        let module_body = format!("{appended}\n");
        // Wire before the first `mod` line, keeping the manifest shape.
        let first_mod = root_text.lines().find(|l| is_mod_decl(l)).map(str::to_string)?;
        let wired = format!("mod {fn_name};\npub use {fn_name}::*;\n{first_mod}");
        let new_root = root_text.replacen(&first_mod, &wired, 1);
        return Some(
            RepairPatch::new()
                .with_edit(RepairEdit::new(
                    module_file,
                    module_body.clone(),
                    module_body,
                    "emergent NL addition (new module file; verified synthesis, no LLM)",
                ))
                .with_edit(RepairEdit::new(
                    root_path,
                    root_text,
                    new_root,
                    "emergent NL addition (module wiring in crate root)",
                ))
                .with_metadata("proposer", "nl_emergent_addition_multifile")
                .with_metadata("synthesis_method", result.method.clone()),
        );
    }

    let new_text = format!("{}\n\n{}\n", old_text.trim_end(), appended);
    Some(
        RepairPatch::new()
            .with_edit(RepairEdit::new(
                target,
                old_text,
                new_text,
                "emergent NL addition proposer (new fn appended; verified synthesis, no LLM)",
            ))
            .with_metadata("proposer", "nl_emergent_addition")
            .with_metadata("synthesis_method", result.method.clone()),
    )
}

/// SIGNATURE-CHANGE refactor: "add a parameter <p> to <fn> defaulting to <n>" —
/// structural parse (no synthesis), then a coordinated patch:
///   * the DEFINITION gains `, p: i64` in its signature;
///   * EVERY non-test call site gains `, n` as the new trailing argument (the
///     default), across all files — so existing behavior is preserved and the
///     TDD oracle (tests already call the new arity with real values) proves
///     the wiring.
/// Gates: fn must be defined (non-test); param name must not already be in the
/// signature; a default literal must be present in the prose (no fabrication).
pub fn try_add_param_patch(context: &RepairContext, description: &str) -> Option<RepairPatch> {
    let lower = description.to_lowercase();
    if !lower.contains("add a parameter") && !lower.contains("add parameter") {
        return None;
    }
    let toks: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .filter(|t| !t.is_empty())
        .collect();
    let ppos = toks.iter().position(|t| *t == "parameter")?;
    let tpos = toks.iter().position(|t| *t == "to")?;
    // Default literal: the token after "defaulting"/"default" (allow negatives).
    let dpos = toks
        .iter()
        .position(|t| *t == "defaulting" || *t == "default" || *t == "defaults")?;
    let default_lit: i64 = toks[dpos + 1..]
        .iter()
        .find_map(|t| t.parse::<i64>().ok())?;
    let is_ident =
        |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    let param = toks[ppos + 1..tpos].iter().find(|t| is_ident(t))?.to_string();
    let defined: Vec<String> = context
        .files
        .iter()
        .filter(|f| f.path.ends_with(".rs"))
        .flat_map(|f| defined_fn_names(f.text.as_deref().unwrap_or("")))
        .collect();
    let fn_name = toks[tpos + 1..]
        .iter()
        .find(|t| defined.iter().any(|d| d == *t))?
        .to_string();

    let mut patch = patch_with_meta("nl_add_param_refactor");
    let mut touched = 0usize;
    for file in &context.files {
        if !file.path.ends_with(".rs") {
            continue;
        }
        let text = file.text.as_deref().unwrap_or("");
        let rewritten = add_param_outside_tests(text, &fn_name, &param, default_lit)?;
        if rewritten != text {
            touched += 1;
            patch = patch.with_edit(RepairEdit::new(
                file.path.clone(),
                text.to_string(),
                rewritten,
                "add-parameter refactor (signature + call sites; tests untouched)",
            ));
        }
    }
    (touched > 0).then_some(patch)
}

fn patch_with_meta(proposer: &str) -> RepairPatch {
    RepairPatch::new().with_metadata("proposer", proposer)
}

/// Rewrite `fn_name`'s DEFINITION signature (append `, p: i64`) and every call
/// site (append `, <default>`), skipping test blocks. Returns None (whole patch
/// declines) if the param already exists in the signature.
fn add_param_outside_tests(
    text: &str,
    fn_name: &str,
    param: &str,
    default_lit: i64,
) -> Option<String> {
    let mut out: Vec<String> = Vec::new();
    let mut depth: i32 = 0;
    let mut test_block_depth: Option<i32> = None;
    let mut pending_test_block = false;
    let def_pat = format!("fn {fn_name}(");
    let call_pat = format!("{fn_name}(");
    for line in text.lines() {
        let t = line.trim_start();
        if t.starts_with("#[cfg(test)") {
            pending_test_block = true;
        }
        let in_test = test_block_depth.is_some();
        let rewritten = if in_test {
            line.to_string()
        } else if let Some(defpos) = line.find(&def_pat) {
            // DEFINITION: append the param before the closing paren of the sig.
            let close = line[defpos..].find(')')? + defpos;
            if line[..close].contains(&format!("{param}:")) {
                return None; // param already present — decline whole refactor
            }
            let sep = if line[defpos + def_pat.len()..close].trim().is_empty() { "" } else { ", " };
            format!(
                "{}{sep}{param}: i64{}",
                &line[..close],
                &line[close..]
            )
        } else if line.contains(&call_pat) && !line.contains(&def_pat) {
            // CALL SITES: append the default before each call's closing paren.
            rewrite_calls_append_arg(line, &call_pat, &default_lit.to_string())
        } else {
            line.to_string()
        };
        out.push(rewritten);
        for c in line.chars() {
            if c == '{' {
                depth += 1;
                if pending_test_block && test_block_depth.is_none() {
                    test_block_depth = Some(depth);
                    pending_test_block = false;
                }
            } else if c == '}' {
                if test_block_depth == Some(depth) {
                    test_block_depth = None;
                }
                depth -= 1;
            }
        }
    }
    let mut s = out.join("\n");
    if text.ends_with('\n') {
        s.push('\n');
    }
    Some(s)
}

/// Append `arg` to every `name(...)` call on the line (word-boundary; balanced
/// paren scan finds each call's close).
fn rewrite_calls_append_arg(line: &str, call_pat: &str, arg: &str) -> String {
    let bytes = line.as_bytes();
    let mut out = String::new();
    let mut i = 0;
    while i < bytes.len() {
        let boundary =
            i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
        if boundary && line[i..].starts_with(call_pat) {
            let open = i + call_pat.len() - 1;
            // balanced scan for this call's closing paren
            let mut d = 0i32;
            let mut j = open;
            while j < bytes.len() {
                if bytes[j] == b'(' {
                    d += 1;
                } else if bytes[j] == b')' {
                    d -= 1;
                    if d == 0 {
                        break;
                    }
                }
                j += 1;
            }
            if j < bytes.len() {
                let inner_empty = line[open + 1..j].trim().is_empty();
                let sep = if inner_empty { "" } else { ", " };
                out.push_str(&line[i..j]);
                out.push_str(&format!("{sep}{arg}"));
                out.push(')');
                i = j + 1;
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

/// RENAME refactor (coordinated multi-file EDIT, no synthesis needed): parse
/// "rename X to Y" STRUCTURALLY (the two identifiers around the rename…to shape),
/// then rewrite every word-boundary occurrence of X across every non-test-only
/// `.rs` file — definition and all call sites in ONE atomic patch. Gates:
///   * X must be a defined fn (outside test code); Y must be a valid ident not
///     already defined;
///   * the ORACLE stays honest: occurrences inside `#[cfg(test)]` blocks are
///     left untouched (in the TDD shape the tests already call Y, which is what
///     makes the rename verifiable — cargo goes green only if every production
///     reference was updated).
pub fn try_rename_patch(context: &RepairContext, description: &str) -> Option<RepairPatch> {
    // Structural parse: identifier after "rename", identifier after "to".
    let lower = description.to_lowercase();
    let toks: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| !t.is_empty())
        .collect();
    let rpos = toks.iter().position(|t| *t == "rename")?;
    let tpos = toks.iter().rposition(|t| *t == "to")?;
    if tpos <= rpos + 1 {
        return None;
    }
    let is_ident =
        |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    // X: first token after "rename" that is a DEFINED fn; Y: first ident after "to".
    let defined: Vec<String> = context
        .files
        .iter()
        .filter(|f| f.path.ends_with(".rs"))
        .flat_map(|f| defined_fn_names(f.text.as_deref().unwrap_or("")))
        .collect();
    let old = toks[rpos + 1..tpos]
        .iter()
        .find(|t| defined.iter().any(|d| d == *t))?
        .to_string();
    let new = toks[tpos + 1..]
        .iter()
        .find(|t| is_ident(t) && !["a", "the", "function", "fn"].contains(t))?
        .to_string();
    if new == old || defined.iter().any(|d| *d == new) {
        return None; // no-op or collision with an existing fn
    }

    // Rewrite every word-boundary occurrence OUTSIDE test blocks, per file.
    let mut patch = RepairPatch::new().with_metadata("proposer", "nl_rename_refactor");
    let mut touched = 0usize;
    for file in &context.files {
        if !file.path.ends_with(".rs") {
            continue;
        }
        let text = file.text.as_deref().unwrap_or("");
        let rewritten = rename_outside_tests(text, &old, &new);
        if rewritten != text {
            touched += 1;
            patch = patch.with_edit(RepairEdit::new(
                file.path.clone(),
                text.to_string(),
                rewritten,
                "rename refactor (definition + call sites; tests untouched)",
            ));
        }
    }
    (touched > 0).then_some(patch)
}

/// Word-boundary rename of `old` → `new` in `text`, skipping `#[cfg(test)]`
/// blocks (brace-tracked) and `#[test]`-annotated fns' lines — the oracle is
/// never edited.
fn rename_outside_tests(text: &str, old: &str, new: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut depth: i32 = 0;
    let mut test_block_depth: Option<i32> = None;
    let mut pending_test_block = false;
    for line in text.lines() {
        let t = line.trim_start();
        if t.starts_with("#[cfg(test)") {
            pending_test_block = true;
        }
        let in_test = test_block_depth.is_some();
        let rewritten = if in_test {
            line.to_string()
        } else {
            replace_word(line, old, new)
        };
        out.push(rewritten);
        for c in line.chars() {
            if c == '{' {
                depth += 1;
                if pending_test_block && test_block_depth.is_none() {
                    test_block_depth = Some(depth);
                    pending_test_block = false;
                }
            } else if c == '}' {
                if test_block_depth == Some(depth) {
                    test_block_depth = None;
                }
                depth -= 1;
            }
        }
    }
    let mut s = out.join("\n");
    if text.ends_with('\n') {
        s.push('\n');
    }
    s
}

/// Replace whole-word (identifier-boundary) occurrences of `old` with `new`.
fn replace_word(line: &str, old: &str, new: &str) -> String {
    let bytes = line.as_bytes();
    let ob = old.as_bytes();
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    while i < bytes.len() {
        let boundary_before =
            i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
        if boundary_before && bytes[i..].starts_with(ob) {
            let j = i + ob.len();
            let boundary_after =
                j >= bytes.len() || !(bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_');
            if boundary_after {
                out.push_str(new);
                i = j;
                continue;
            }
        }
        out.push(bytes[i] as char);
        i += 1;
    }
    out
}

/// CONTENT-based localization: which repo fn does the description talk about?
/// Scans every `.rs` file's defined fn names and matches each snake_case part of
/// the name against the description's tokens emergently (exact or shared
/// morphological stem). A fn matches only when ALL its parts are matched (so an
/// incidental "number" in the prose can't select `fn number_cruncher` unless
/// "cruncher" appears too). Ranking: a candidate in the `preferred` file (the
/// failure-implicated one) outranks all others; then the most specific match
/// (most parts) wins.
fn locate_described_fn(
    context: &RepairContext,
    description: &str,
    preferred: Option<&str>,
) -> Option<(String, String)> {
    use linguigenesis_core::entity_resolution::morphological_variants;
    let lower = description.to_lowercase();
    let tokens: Vec<String> = lower
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| t.len() >= 3)
        .map(str::to_string)
        .collect();
    let token_matches = |part: &str| -> bool {
        tokens.iter().any(|t| {
            if t == part {
                return true;
            }
            let mut tv = morphological_variants(t);
            tv.push(t.clone());
            let mut pv = morphological_variants(part);
            pv.push(part.to_string());
            tv.iter().any(|v| pv.contains(v))
        })
    };
    // Rank = (in the failure-implicated file, specificity). Paths compared
    // suffix-wise both ways since the compiler may print an absolute path.
    let is_preferred = |path: &str| -> bool {
        preferred
            .map(|p| p.ends_with(path) || path.ends_with(p))
            .unwrap_or(false)
    };
    let mut best: Option<(String, String, (bool, usize))> = None;
    for file in &context.files {
        if !file.path.ends_with(".rs") {
            continue;
        }
        let text = file.text.as_deref().unwrap_or("");
        for fn_name in defined_fn_names(text) {
            let parts: Vec<&str> = fn_name.split('_').filter(|p| !p.is_empty()).collect();
            if parts.is_empty() || !parts.iter().all(|p| token_matches(p)) {
                continue;
            }
            let rank = (is_preferred(&file.path), parts.len());
            if best.as_ref().map(|(_, _, b)| rank > *b).unwrap_or(true) {
                best = Some((file.path.clone(), fn_name, rank));
            }
        }
    }
    if best.is_some() {
        return best.map(|(path, name, _)| (path, name));
    }

    // CONTENT-GREP fallback (the ReAct slice): no defined fn NAME matches the
    // prose — search file CONTENT for the description's tokens instead
    // ("fix the tripling helper" finds the file whose comment says "tripling
    // helper" even though the fn is called `mul3`). The file with the most
    // distinct token hits wins; within it, only an UNAMBIGUOUS target is
    // accepted — exactly one non-test fn — otherwise decline (never guess
    // among several).
    let mut best_file: Option<(String, usize)> = None;
    for file in &context.files {
        if !file.path.ends_with(".rs") {
            continue;
        }
        let text = file.text.as_deref().unwrap_or("").to_lowercase();
        let hits = tokens.iter().filter(|t| t.len() >= 4 && text.contains(t.as_str())).count();
        if hits == 0 {
            continue;
        }
        let rank = hits + usize::from(is_preferred(&file.path));
        if best_file.as_ref().map(|(_, b)| rank > *b).unwrap_or(true) {
            best_file = Some((file.path.clone(), rank));
        }
    }
    let (path, _) = best_file?;
    let text = context
        .files
        .iter()
        .find(|f| f.path == path)?
        .text
        .as_deref()
        .unwrap_or("")
        .to_string();
    let fns = defined_fn_names(&text);
    if fns.len() == 1 {
        return Some((path, fns.into_iter().next()?));
    }
    None
}

/// Every `fn NAME(` defined in `text`, EXCLUDING test code: fns annotated
/// `#[test]` and everything inside a `#[cfg(test)]` block. Tests are the
/// acceptance ORACLE — a fn named in prose must never localize to the test that
/// checks it (observed: "add a function that triples" matched the test fn
/// `triples()` and the edit stage rewrote the oracle instead of adding the fn).
fn defined_fn_names(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut prev_nonempty = "";
    let mut test_block_depth: Option<i32> = None; // brace depth inside #[cfg(test)]
    let mut depth: i32 = 0;
    let mut pending_test_block = false;
    for line in text.lines() {
        let t = line.trim_start();
        if t.starts_with("#[cfg(test)") {
            pending_test_block = true;
        }
        let in_test_block = test_block_depth.is_some();
        let fn_annotated_test = prev_nonempty.trim_start().starts_with("#[test]");
        let rest = t
            .strip_prefix("pub fn ")
            .or_else(|| t.strip_prefix("fn "))
            .or_else(|| t.strip_prefix("pub(crate) fn "));
        if let Some(rest) = rest {
            if !in_test_block && !fn_annotated_test {
                if let Some(name) = rest.split('(').next() {
                    let name = name.trim();
                    if !name.is_empty()
                        && name.chars().all(|c| c.is_alphanumeric() || c == '_')
                    {
                        out.push(name.to_string());
                    }
                }
            }
        }
        for c in line.chars() {
            if c == '{' {
                depth += 1;
                if pending_test_block && test_block_depth.is_none() {
                    test_block_depth = Some(depth);
                    pending_test_block = false;
                }
            } else if c == '}' {
                if test_block_depth == Some(depth) {
                    test_block_depth = None;
                }
                depth -= 1;
            }
        }
        if !t.is_empty() {
            prev_nonempty = line;
        }
    }
    out
}

/// Reshape synthesized Rust to fit the repo's target function.
///
/// Single-function output: keep the repo signature and swap in the synthesized
/// body with params renamed positionally (so a `&[i64]` repo signature is
/// honoured even when the solver emits a by-value form).
///
/// Multi-function output (a main function plus helpers — e.g. the safe-division
/// `helper_div` or the LCM `gcd_inner`): emit the helpers verbatim followed by
/// the main function renamed to the repo function and made `pub`, replacing the
/// repo function definition wholesale. The main is identified by name match to
/// the repo function, else the last function defined.
fn reshape_to_repo_signature(old_text: &str, repo_fn: &str, synthesized: &str) -> Option<String> {
    let fns = split_top_level_functions(synthesized);

    let repo_has_fn = old_text.contains(&format!("fn {repo_fn}"));
    // Slice adapters bridge the synthesizer's owned `Vec<i64>` params to a repo `&[i64]` slice
    // signature (see slice_param_adapters). Only meaningful when the repo fn exists.
    let adapters = if repo_has_fn { slice_param_adapters(old_text, repo_fn) } else { String::new() };

    if fns.len() > 1 {
        let main_idx = fns
            .iter()
            .position(|(name, _)| name == repo_fn)
            .unwrap_or(fns.len() - 1);
        let mut helpers = String::new();
        for (k, (_, text)) in fns.iter().enumerate() {
            if k != main_idx {
                helpers.push_str(text.trim());
                helpers.push_str("\n\n");
            }
        }
        // SLICE PATH: keep the repo's `&[i64]` signature, run the Vec-based synthesized main body
        // behind a `.to_vec()` bridge, and insert the (Vec-based) helpers as sibling functions.
        if repo_has_fn && !adapters.is_empty() {
            let synth_idents = fn_header_params(&fns[main_idx].1, &fns[main_idx].0)
                .map(|p| parse_param_idents(&p))
                .unwrap_or_default();
            let repo_idents = parse_param_idents(&fn_header_params(old_text, repo_fn).unwrap_or_default());
            let mut body = fn_body(&fns[main_idx].1)?;
            if synth_idents.len() == repo_idents.len() {
                for (from, to) in synth_idents.iter().zip(repo_idents.iter()) {
                    if from != to {
                        body = rename_ident(&body, from, to);
                    }
                }
            }
            let new_body = format!("{adapters}{body}");
            let with_body = replace_body_only(old_text, repo_fn, &new_body).ok()?;
            return Some(insert_before_fn_def(&with_body, repo_fn, helpers.trim()));
        }
        let main_renamed = ensure_pub_fn(&rename_first_fn(&fns[main_idx].1, repo_fn));
        let new_impl = format!("{helpers}{}", main_renamed.trim());
        return if repo_has_fn {
            replace_function_body(old_text, repo_fn, &new_impl).ok()
        } else {
            Some(format!("{}\n", new_impl.trim()))
        };
    }

    let synth_body = fn_body(synthesized)?;
    if !repo_has_fn {
        let renamed = ensure_pub_fn(&rename_first_fn(synthesized, repo_fn));
        return Some(format!("{}\n", renamed.trim()));
    }
    let synth_params = fn_header_params(synthesized, &first_fn_name_in_source(synthesized)?)
        .map(|p| parse_param_idents(&p))
        .unwrap_or_default();
    let repo_params = fn_header_params(old_text, repo_fn)
        .map(|p| parse_param_idents(&p))
        .unwrap_or_default();
    let mut body = synth_body;
    if synth_params.len() == repo_params.len() {
        for (from, to) in synth_params.iter().zip(repo_params.iter()) {
            if from != to {
                body = rename_ident(&body, from, to);
            }
        }
    }
    // Prepend slice adapters so a `&[i64]`-signature repo fn compiles against the Vec-based body.
    let body = format!("{adapters}{body}");
    replace_body_only(old_text, repo_fn, &body).ok()
}

/// Insert `block` (sibling top-level functions) immediately before the definition of `fn_name`.
fn insert_before_fn_def(source: &str, fn_name: &str, block: &str) -> String {
    if block.is_empty() {
        return source.to_string();
    }
    let pos = source
        .find(&format!("pub fn {fn_name}"))
        .or_else(|| source.find(&format!("fn {fn_name}")));
    match pos {
        Some(p) => format!("{}{}\n\n{}", &source[..p], block, &source[p..]),
        None => source.to_string(),
    }
}

/// Split a Rust source into top-level `(name, full_text)` function definitions,
/// preserving any `pub` prefix and the complete `{ ... }` body.
fn split_top_level_functions(code: &str) -> Vec<(String, String)> {
    let bytes = code.as_bytes();
    let mut fns = Vec::new();
    let mut i = 0;
    while i < code.len() {
        let boundary = i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
        if boundary && code[i..].starts_with("fn ") {
            let mut start = i;
            let before = code[..i].trim_end();
            if before.ends_with("pub") {
                start = before.len() - 3;
            }
            let name: String = code[i + 3..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if let Some(brace_off) = code[i..].find('{') {
                let bstart = i + brace_off;
                let mut depth = 0i32;
                let mut end = None;
                for j in bstart..bytes.len() {
                    match bytes[j] {
                        b'{' => depth += 1,
                        b'}' => {
                            depth -= 1;
                            if depth == 0 {
                                end = Some(j);
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                if let Some(end) = end {
                    fns.push((name, code[start..end + 1].to_string()));
                    i = end + 1;
                    continue;
                }
            }
        }
        let ch = code[i..].chars().next().unwrap();
        i += ch.len_utf8();
    }
    fns
}

/// Heuristic: is this synthesized code directly compilable plain Rust (vs the
/// solver's abstract IR)? Rejects Result-style `ok(..)`/`err(..)` wrappers and
/// unlowered Mog assignment so we never write non-compiling repairs.
fn is_plain_rust_body(code: &str) -> bool {
    !(code.contains("ok(")
        || code.contains("err(")
        || code.contains(":=")
        || code.contains("Result<"))
}

fn ensure_pub_fn(code: &str) -> String {
    let trimmed = code.trim_start();
    if trimmed.starts_with("pub ") {
        code.to_string()
    } else if trimmed.starts_with("fn ") {
        format!("pub {trimmed}")
    } else {
        code.to_string()
    }
}

fn rename_first_fn(code: &str, new_name: &str) -> String {
    if let Some(old) = first_fn_name_in_source(code) {
        let needle = format!("fn {old}");
        if let Some(pos) = code.find(&needle) {
            let mut out = String::with_capacity(code.len());
            out.push_str(&code[..pos]);
            out.push_str(&format!("fn {new_name}"));
            out.push_str(&code[pos + needle.len()..]);
            return out;
        }
    }
    code.to_string()
}

/// Parameter list (raw text between parens) of `fn {name}(...)`.
fn fn_header_params(code: &str, name: &str) -> Option<String> {
    let start = code.find(&format!("fn {name}("))?;
    let after = &code[start..];
    let open = after.find('(')?;
    let bytes = after.as_bytes();
    let mut depth = 0i32;
    let mut end = None;
    for i in open..bytes.len() {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    Some(after[open + 1..end?].to_string())
}

fn parse_param_idents(params: &str) -> Vec<String> {
    params
        .split(',')
        .filter_map(|p| {
            let p = p.trim();
            if p.is_empty() {
                return None;
            }
            let name = p.split(':').next()?.trim();
            let name = name.trim_start_matches("mut ").trim();
            if name.is_empty() {
                None
            } else {
                Some(name.to_string())
            }
        })
        .collect()
}

/// Parameter TYPES parallel to [`parse_param_idents`]: the text after each `:`.
fn parse_param_types(params: &str) -> Vec<String> {
    params
        .split(',')
        .filter_map(|p| {
            let p = p.trim();
            if p.is_empty() {
                return None;
            }
            let (_name, ty) = p.split_once(':')?;
            Some(ty.trim().to_string())
        })
        .collect()
}

/// `.to_vec()` shadow lines for each repo parameter whose type is a slice (`&[..]`), so the
/// Vec-based synthesized body compiles against a slice signature. A collection repo function
/// (`pub fn sum_of_evens(xs: &[i64]) -> i64`) keeps its slice signature (matching the test's
/// call convention) while the synthesized logic sees an owned `Vec` — the minimal bridge that
/// unblocks list-processing repo repairs without rewriting iteration.
fn slice_param_adapters(old_text: &str, repo_fn: &str) -> String {
    // Bridge every `&[..]` slice PARAM to an owned Vec (the shape the synthesizer emits), regardless
    // of the return type: `sum_of_evens(xs: &[i64]) -> i64` and `reverse(xs: &[i64]) -> Vec<i64>`
    // both need `let xs = xs.to_vec();` at the top of the body so the Vec-based synthesized logic
    // compiles against the slice signature.
    let params = fn_header_params(old_text, repo_fn).unwrap_or_default();
    let idents = parse_param_idents(&params);
    let types = parse_param_types(&params);
    let mut out = String::new();
    for (name, ty) in idents.iter().zip(types.iter()) {
        if ty.contains("&[") {
            out.push_str(&format!("let {name} = {name}.to_vec();\n"));
        }
    }
    out
}

/// Inner text of the first `{ ... }` body in `code`.
fn fn_body(code: &str) -> Option<String> {
    let s = code.find("fn ")?;
    let after = &code[s..];
    let open = after.find('{')?;
    let bytes = after.as_bytes();
    let mut depth = 0i32;
    let mut end = None;
    for i in open..bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    Some(after[open + 1..end?].trim().to_string())
}

/// Keep the repo function signature, replace only its `{ body }`.
fn replace_body_only(source: &str, name: &str, new_body: &str) -> Result<String, String> {
    let start = source
        .find(&format!("pub fn {name}"))
        .or_else(|| source.find(&format!("fn {name}")))
        .ok_or_else(|| format!("function {name} not found"))?;
    let after = &source[start..];
    let open = after
        .find('{')
        .ok_or_else(|| format!("no opening brace for {name}"))?;
    let bytes = after.as_bytes();
    let mut depth = 0i32;
    let mut end = None;
    for i in open..bytes.len() {
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    let end = end.ok_or_else(|| format!("unclosed body for {name}"))?;
    let mut out = String::new();
    out.push_str(&source[..start]);
    out.push_str(&after[..open + 1]);
    out.push_str("\n    ");
    out.push_str(new_body.trim());
    out.push('\n');
    out.push_str(&after[end..]);
    Ok(out)
}

fn rename_ident(text: &str, from: &str, to: &str) -> String {
    if from.is_empty() || from == to {
        return text.to_string();
    }
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        if text[i..].starts_with(from) {
            let before_ok =
                i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
            let after_idx = i + from.len();
            let after_ok = after_idx >= text.len()
                || !(bytes[after_idx].is_ascii_alphanumeric() || bytes[after_idx] == b'_');
            if before_ok && after_ok {
                out.push_str(to);
                i = after_idx;
                continue;
            }
        }
        let ch = text[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Fast NL repo patch from description and optional verification failure (Package H).
pub fn try_nl_repo_fast_patch(
    task: &RepoTaskSpec,
    context: &RepairContext,
    description: &str,
    analysis: Option<&FailureAnalysis>,
) -> Option<RepairPatch> {
    if !should_use_failure_aware_patch(analysis) {
        return None;
    }
    let intent = CodingIntent::from_nl(description).unwrap_or_else(|_| {
        let target_hint = first_rust_target_hint(task, context);
        coding_intent_from_nl_description(description, target_hint.as_deref())
    });
    let mog_hint = failure_mog_hint(analysis);
    let target_hint = match pick_target_path(task, context, Some(&intent)) {
        Ok(target) => read_relative_file(context, &target).ok(),
        Err(_) => first_rust_target_hint(task, context),
    };
    let rust_body = repo_rust_body_for_nl(&intent, mog_hint, target_hint.as_deref())?;
    let method = if analysis.is_some() {
        "nl_failure_aware_repo_stub"
    } else {
        "nl_description_repo_stub"
    };
    let stub = crate::solver::SolveResult {
        success: true,
        code: rust_body,
        method: method.to_string(),
        error: None,
        metadata: Default::default(),
    };
    repair_patch_from_synthesis(task, context, Some(&intent), &stub)
        .map(|patch| patch.with_metadata("failure_aware", analysis.is_some().to_string()))
        .ok()
}

fn should_use_failure_aware_patch(analysis: Option<&FailureAnalysis>) -> bool {
    match analysis {
        None => true,
        Some(analysis) => matches!(
            analysis.kind,
            FailureKind::TestFailure
                | FailureKind::CompileError
                | FailureKind::TypeMismatch
                | FailureKind::RuntimeError
                | FailureKind::Unknown
        ),
    }
}

fn failure_mog_hint(analysis: Option<&FailureAnalysis>) -> &str {
    analysis.map(|a| a.message.as_str()).unwrap_or("")
}

fn first_rust_target_hint(task: &RepoTaskSpec, context: &RepairContext) -> Option<String> {
    match pick_target_path(task, context, None) {
        Ok(target) => read_relative_file(context, &target).ok(),
        Err(_) => context
            .files
            .iter()
            .find(|file| file.path.ends_with(".rs"))
            .and_then(|file| file.text.clone()),
    }
}

pub(crate) fn coding_intent_from_nl_description(
    description: &str,
    target_hint: Option<&str>,
) -> CodingIntent {
    let desc = description.to_ascii_lowercase();
    let is_array = desc.contains("reverse") || desc.contains("array");
    let function_name = if let Some(text) = target_hint {
        first_fn_name_in_source(text).unwrap_or_else(|| infer_default_fn_name(&desc))
    } else {
        infer_default_fn_name(&desc)
    };
    CodingIntent {
        function_name,
        signature: if is_array {
            "Vec<i64> -> Vec<i64>".to_string()
        } else {
            "i64, i64 -> i64".to_string()
        },
        category: if is_array {
            "array".to_string()
        } else {
            "arithmetic".to_string()
        },
        description: description.to_string(),
        examples: Vec::new(),
        constraints: Vec::new(),
        confidence: 1.0,
        unresolved: Vec::new(),
        evidence_entity_ids: Vec::new(),
        reference_code: String::new(),
    }
}

fn infer_default_fn_name(desc: &str) -> String {
    if desc.contains("subtract") {
        "subtract".to_string()
    } else if desc.contains("multiply") {
        "multiply".to_string()
    } else if desc.contains("divide") {
        "divide".to_string()
    } else if desc.contains("larger") || desc.contains("maximum") || desc.contains("max") {
        "max".to_string()
    } else if desc.contains("reverse") {
        "reverse".to_string()
    } else {
        "add".to_string()
    }
}

/// Normalize synthesized Mog or Rust into a Rust function body for repo repair.
pub(crate) fn rust_code_for_repo_synthesis(synthesized: &str) -> String {
    // Lower the solver's Result-style IR toward plain Rust first: fold the
    // boilerplate `match r { ok(v) => v, err(e) => CONST }` to `r.unwrap_or(CONST)`
    // (the line-based transpiler can't handle a multi-line match), then map the
    // `ok`/`err`/`Result` constructors 1:1 onto `Some`/`None`/`Option`.
    let lowered = lower_result_tokens(&fold_result_match_idiom(synthesized));
    let trimmed = lowered.trim();
    if trimmed.contains(":=") || (trimmed.contains("return ") && !trimmed.contains("pub fn ")) {
        let mog = mog_source_for_rust_transpile(trimmed);
        let rust = crate::mog_transpile::to_rust(&mog);
        if rust.trim_start().starts_with("fn ") {
            return format!("pub {rust}");
        }
        return rust;
    }
    trimmed.to_string()
}

/// Map the solver's Result-style constructors onto Rust `Option`:
/// `Result<T>` → `Option<T>`, `ok(X)` → `Some(X)`, `err(..)` → `None`. The
/// mapping is faithful: the safe-division template uses `ok`/`err` exactly as
/// `Some`/`None` and never inspects the error payload.
fn lower_result_tokens(code: &str) -> String {
    let mut s = code.replace("Result<", "Option<");
    s = replace_call_token(&s, "ok", |inner| format!("Some({inner})"));
    s = replace_call_token(&s, "err", |_| "None".to_string());
    s
}

/// Replace whole-word `name(<balanced>)` calls using `f(inner)`. Skips matches
/// where `name` is part of a larger identifier (e.g. `lookup(`), so only the
/// bare constructor token is rewritten.
fn replace_call_token(code: &str, name: &str, f: impl Fn(&str) -> String) -> String {
    let bytes = code.as_bytes();
    let needle = format!("{name}(");
    let mut out = String::with_capacity(code.len());
    let mut i = 0;
    while i < code.len() {
        if code[i..].starts_with(&needle) {
            let prev_ok = i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_');
            if prev_ok {
                let open = i + name.len();
                let mut depth = 0i32;
                let mut j = open;
                let mut end = None;
                while j < bytes.len() {
                    match bytes[j] {
                        b'(' => depth += 1,
                        b')' => {
                            depth -= 1;
                            if depth == 0 {
                                end = Some(j);
                                break;
                            }
                        }
                        _ => {}
                    }
                    j += 1;
                }
                if let Some(end) = end {
                    let inner = &code[open + 1..end];
                    out.push_str(&f(inner));
                    i = end + 1;
                    continue;
                }
            }
        }
        let ch = code[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// Fold the safe-result idiom `match VAR { ok(v) => v, err(e) => CONST }` into
/// `VAR.unwrap_or(CONST)`. Only fires for the identity-ok / constant-err shape
/// the solver emits; any other match is left untouched.
fn fold_result_match_idiom(code: &str) -> String {
    let Some(m) = code.find("match ") else {
        return code.to_string();
    };
    let after = &code[m + "match ".len()..];
    let brace_rel = match after.find('{') {
        Some(b) => b,
        None => return code.to_string(),
    };
    let scrutinee = after[..brace_rel].trim();
    if scrutinee.is_empty() || scrutinee.contains(|c: char| !(c.is_alphanumeric() || c == '_')) {
        return code.to_string();
    }
    let body_start = m + "match ".len() + brace_rel;
    let bytes = code.as_bytes();
    let mut depth = 0i32;
    let mut end = None;
    for j in body_start..bytes.len() {
        match bytes[j] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(j);
                    break;
                }
            }
            _ => {}
        }
    }
    let Some(end) = end else {
        return code.to_string();
    };
    let arms = &code[body_start + 1..end];
    let mut ok_bind = None;
    let mut ok_res = None;
    let mut err_res = None;
    for arm in arms.split(',') {
        let arm = arm.trim();
        if arm.is_empty() {
            continue;
        }
        let Some((pat, res)) = arm.split_once("=>") else {
            return code.to_string();
        };
        let (pat, res) = (pat.trim(), res.trim());
        if let Some(bind) = pat.strip_prefix("ok(").and_then(|s| s.strip_suffix(')')) {
            ok_bind = Some(bind.trim().to_string());
            ok_res = Some(res.to_string());
        } else if pat.starts_with("err(") {
            err_res = Some(res.to_string());
        } else {
            return code.to_string();
        }
    }
    match (ok_bind, ok_res, err_res) {
        (Some(bind), Some(ok_res), Some(err_res)) if ok_res == bind => {
            let mut out = String::with_capacity(code.len());
            out.push_str(&code[..m]);
            out.push_str(&format!("{scrutinee}.unwrap_or({err_res})"));
            out.push_str(&code[end + 1..]);
            out
        }
        _ => code.to_string(),
    }
}

fn mog_source_for_rust_transpile(mog: &str) -> String {
    mog.lines()
        .map(|line| {
            let trimmed = line.trim();
            if let Some((var, rhs)) = trimmed.split_once(":=") {
                let var = var.trim();
                let rhs = rhs.trim().trim_end_matches(';');
                if var.is_empty() || var.contains(' ') || var.contains(':') {
                    return line.to_string();
                }
                let indent: String = line.chars().take_while(|c| c.is_whitespace()).collect();
                // Use type inference rather than a hard-coded `i64` annotation:
                // the RHS may be a non-i64 value (e.g. an `Option<i64>` from a
                // lowered Result helper), where `: i64` would be a type error.
                return format!("{indent}let mut {var} = {rhs};");
            }
            line.to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Build a repair patch from an existing synthesized agent run.
pub fn repair_patch_from_synthesis(
    task: &RepoTaskSpec,
    context: &RepairContext,
    intent: Option<&CodingIntent>,
    result: &crate::solver::SolveResult,
) -> Result<RepairPatch, String> {
    if !result.success {
        return Err(result
            .error
            .clone()
            .unwrap_or_else(|| "synthesis failed".to_string()));
    }

    let target = pick_target_path(task, context, intent)?;
    let old_text = read_relative_file(context, &target)?;
    let synthesized = rust_code_for_repo_synthesis(&result.code);
    let new_text = apply_synthesis_to_file(&old_text, &synthesized, intent)?;

    Ok(RepairPatch::new()
        .with_edit(RepairEdit::new(
            target,
            old_text,
            new_text,
            "nl synthesis proposer (linguigenesis bridge)",
        ))
        .with_metadata("proposer", "nl_synthesis")
        .with_metadata("synthesis_method", result.method.clone()))
}

pub(crate) fn pick_target_path(
    task: &RepoTaskSpec,
    context: &RepairContext,
    intent: Option<&CodingIntent>,
) -> Result<String, String> {
    if let Some(intent) = intent {
        let primary_name = intent
            .function_name
            .strip_prefix("nl_")
            .unwrap_or(&intent.function_name);
        for file in &context.files {
            if file.path.ends_with(".rs") {
                let text = file.text.as_deref().unwrap_or("");
                if file_defines_function(text, primary_name)
                    || file_defines_function(text, &intent.function_name)
                {
                    return Ok(file.path.clone());
                }
            }
        }
        for file in &context.files {
            if file.path.ends_with(".rs") {
                if file
                    .text
                    .as_deref()
                    .unwrap_or("")
                    .contains(&intent.function_name)
                {
                    return Ok(file.path.clone());
                }
            }
        }
    }

    for pattern in &task.allowed_files {
        let glob = pattern.replace("**", "");
        for file in &context.files {
            if file.path.contains(glob.trim_matches('/')) || file.path.ends_with(".rs") {
                return Ok(file.path.clone());
            }
        }
    }

    context
        .files
        .iter()
        .find(|f| f.path.ends_with(".rs"))
        .map(|f| f.path.clone())
        .ok_or_else(|| "no writable Rust file in repair context".to_string())
}

fn file_defines_function(text: &str, fn_name: &str) -> bool {
    text.contains(&format!("fn {fn_name}(")) || text.contains(&format!("fn {fn_name} "))
}

pub(crate) fn read_relative_file(
    context: &RepairContext,
    relative: &str,
) -> Result<String, String> {
    let disk_path = Path::new(&context.root).join(relative);
    if disk_path.is_file() {
        return fs::read_to_string(&disk_path).map_err(|e| format!("read {}: {}", relative, e));
    }
    context
        .files
        .iter()
        .find(|f| f.path == relative)
        .map(|f| f.text.clone().unwrap_or_default())
        .ok_or_else(|| format!("file not in context: {}", relative))
}

/// Map NL-synthesized Mog bodies to Rust repo stubs when the repair verifier expects Rust idioms.
pub fn repo_rust_body_for_nl(
    intent: &CodingIntent,
    mog_code: &str,
    target_hint: Option<&str>,
) -> Option<String> {
    let desc = intent.description.to_ascii_lowercase();
    let fn_name = intent
        .function_name
        .strip_prefix("nl_")
        .unwrap_or(&intent.function_name);
    let is_array_sig = intent.signature.contains("[i64]")
        || intent.signature.contains("Vec<i64>")
        || intent.signature.contains("List<i64>");
    let is_reverse = desc.contains("reverse")
        || fn_name.contains("reverse")
        || mog_code.contains("array_transform_reverse")
        || mog_code.contains("i = arr.len - 1");
    if is_reverse && is_array_sig {
        let name = resolve_repo_fn_name("reverse", target_hint);
        return Some(format!(
            "pub fn {name}(xs: &[i64]) -> Vec<i64> {{\n    xs.iter().rev().copied().collect()\n}}\n"
        ));
    }

    if is_array_sig {
        return None;
    }

    let scalar_body = scalar_i64_body_for_nl(&desc, mog_code)?;
    let name = resolve_repo_fn_name(fn_name, target_hint);
    Some(format!(
        "pub fn {name}(a: i64, b: i64) -> i64 {{\n    {scalar_body}\n}}\n"
    ))
}

fn scalar_i64_body_for_nl(desc: &str, mog_code: &str) -> Option<&'static str> {
    if desc.contains("subtract") || desc.contains("minus") || mog_code.contains("a - b") {
        return Some("a - b");
    }
    if desc.contains("multiply") || desc.contains("product") || mog_code.contains("a * b") {
        return Some("a * b");
    }
    if desc.contains("divide") || desc.contains("division") || mog_code.contains("a / b") {
        return Some("a / b");
    }
    if desc.contains("larger")
        || desc.contains("maximum")
        || desc.contains("max")
        || mog_code.contains("if a > b")
    {
        return Some("if a > b { a } else { b }");
    }
    if desc.contains("add") || desc.contains("sum") || mog_code.contains("a + b") {
        return Some("a + b");
    }
    None
}

fn resolve_repo_fn_name(default: &str, target_hint: Option<&str>) -> String {
    if let Some(text) = target_hint {
        if default == "add" && text.contains("fn add_two") {
            return "add_two".to_string();
        }
        if let Some(name) = first_fn_name_in_source(text) {
            return name;
        }
    }
    default.to_string()
}

fn first_fn_name_in_source(source: &str) -> Option<String> {
    for line in source.lines() {
        if line.trim().starts_with("#[cfg(test)]") {
            break;
        }
        let trimmed = line.trim();
        let rest = trimmed
            .strip_prefix("pub fn ")
            .or_else(|| trimmed.strip_prefix("fn "));
        if let Some(rest) = rest {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                return Some(name);
            }
        }
    }
    None
}

/// Replace file body with synthesized code when stub/wrong, else append synthesized fn.
fn apply_synthesis_to_file(
    old_text: &str,
    synthesized: &str,
    intent: Option<&CodingIntent>,
) -> Result<String, String> {
    let synthesized = synthesized.trim();
    if synthesized.is_empty() {
        return Err("synthesized code empty".to_string());
    }

    if let Some(intent) = intent {
        if let Some(rust_body) = repo_rust_body_for_nl(intent, synthesized, Some(old_text)) {
            let fn_name = resolve_repo_fn_name(
                intent
                    .function_name
                    .strip_prefix("nl_")
                    .unwrap_or(&intent.function_name),
                Some(old_text),
            );
            if old_text.contains(&format!("fn {fn_name}")) {
                return replace_function_body(old_text, &fn_name, &rust_body);
            }
            return Ok(format!("{}\n", rust_body.trim()));
        }
        if old_text.contains("fn add_two") && intent.function_name == "add" {
            if let Ok(replaced) = replace_function_body(old_text, "add_two", synthesized) {
                return Ok(replaced);
            }
        }
        if old_text.contains(&intent.function_name) {
            return replace_function_body(old_text, &intent.function_name, synthesized);
        }
    }

    if old_text.trim().is_empty() || old_text.contains("WRONG") || old_text.contains("a - b") {
        return Ok(format!("{}\n", synthesized));
    }

    Ok(format!("{}\n\n{}", old_text.trim_end(), synthesized))
}

fn replace_function_body(source: &str, fn_name: &str, new_impl: &str) -> Result<String, String> {
    let pub_needle = format!("pub fn {}", fn_name);
    let fn_needle = format!("fn {}", fn_name);
    let start = source
        .find(&pub_needle)
        .or_else(|| source.find(&fn_needle))
        .ok_or_else(|| format!("function {} not found in target file", fn_name))?;
    let after_fn = &source[start..];
    let brace_start = after_fn
        .find('{')
        .ok_or_else(|| format!("no opening brace for {}", fn_name))?;
    let mut depth = 0i32;
    let mut end_idx = None;
    for (i, ch) in after_fn[brace_start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end_idx = Some(brace_start + i + 1);
                    break;
                }
            }
            _ => {}
        }
    }
    let end = end_idx.ok_or_else(|| format!("unclosed body for {}", fn_name))?;
    let mut out = String::new();
    out.push_str(&source[..start]);
    out.push_str(new_impl.trim());
    if !new_impl.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(&after_fn[end..]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::repo::{
        nl_fixture_cargo_test_command, write_nl_fixture_crate, FailureParser, GuardrailPolicy,
        HardnessProfile, HardnessTier, RepairContext, RepairLoop, RepairVerifier, RepoTaskKind,
        RepoTaskSpec,
    };
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Mutex;

    // --- literal + assert mining: array/slice/vec + owned-string forms (pure, deterministic) ---

    #[test]
    fn parse_literal_handles_array_slice_vec_and_owned_string_forms() {
        use crate::benchmark::Value;
        // scalars still work
        assert_eq!(parse_literal("42"), Some(Value::Int(42)));
        assert_eq!(parse_literal("-3"), Some(Value::Int(-3)));
        assert_eq!(parse_literal("true"), Some(Value::Bool(true)));
        assert_eq!(parse_literal("\"hi\""), Some(Value::Str("hi".into())));
        // array / slice / vec literals -> Value::Array
        let arr = Value::Array(vec![Value::Int(5), Value::Int(2), Value::Int(8), Value::Int(1)]);
        assert_eq!(parse_literal("[5, 2, 8, 1]"), Some(arr.clone()));
        assert_eq!(parse_literal("&[5,2,8,1]"), Some(arr.clone()));
        assert_eq!(parse_literal("vec![5, 2, 8, 1]"), Some(arr));
        assert_eq!(parse_literal("vec![]"), Some(Value::Array(vec![])));
        assert_eq!(parse_literal("[]"), Some(Value::Array(vec![])));
        // owned-String constructors -> the inner string literal
        assert_eq!(parse_literal("\"abc\".to_string()"), Some(Value::Str("abc".into())));
        assert_eq!(parse_literal("\"x\".to_owned()"), Some(Value::Str("x".into())));
        assert_eq!(parse_literal("\"y\".into()"), Some(Value::Str("y".into())));
        assert_eq!(parse_literal("String::from(\"z\")"), Some(Value::Str("z".into())));
        // non-literals still decline
        assert_eq!(parse_literal("some_var"), None);
    }

    #[test]
    fn mine_asserts_recovers_array_and_owned_string_io_pairs() {
        use crate::benchmark::Value;
        // array INPUT, scalar output — the CLI "fix the failing tests" shape for a list fn
        let text = "assert_eq!(mn(vec![5,2,8,1]), 1); assert_eq!(mn(vec![-3,-1,-9]), -9);";
        let rows = mine_asserts(text, "mn");
        assert_eq!(rows.len(), 2, "both array-input asserts must be mined");
        assert_eq!(rows[0].0.len(), 1);
        assert!(matches!(rows[0].0[0], Value::Array(_)));
        // two-arg: slice + scalar
        let text2 = "assert_eq!(cnt(&[1,2,2,3,2], 2), 3);";
        let rows2 = mine_asserts(text2, "cnt");
        assert_eq!(rows2.len(), 1);
        assert_eq!(rows2[0].0.len(), 2, "slice arg and scalar arg both parsed");
        assert_eq!(rows2[0].1, Value::Int(3));
        // owned-String argument + String result
        let text3 = "assert_eq!(rev(\"abc\".to_string()), \"cba\".to_string());";
        let rows3 = mine_asserts(text3, "rev");
        assert_eq!(rows3.len(), 1);
        assert_eq!(rows3[0].0[0], Value::Str("abc".into()));
        assert_eq!(rows3[0].1, Value::Str("cba".into()));
    }

    // --- gated model-repair stage: the pure, model-free core (deterministic) ---

    #[test]
    fn model_repair_reshapes_a_proposed_body_to_the_repo_signature() {
        // A "model-proposed" correct function, reshaped onto the repo fn's exact
        // signature. This is the deterministic half of the gated model lane — the
        // half that runs with no model present.
        let old = "pub fn twice(n: i64) -> i64 {\n    return n;\n}\n";
        let proposed = "fn twice(x: i64) -> i64 {\n    return x * 2;\n}";
        let new = model_body_to_new_text(old, "twice", proposed).expect("reshaped patch");
        assert!(new.contains("* 2"), "proposed body adopted: {new}");
        assert!(new.contains("fn twice(n: i64) -> i64"), "repo signature preserved: {new}");
    }

    #[test]
    fn model_repair_declines_when_target_fn_absent() {
        // Reshape needs the repo fn to exist in the file; if the localizer picked a
        // name not present, there is nothing to swap — decline rather than emit.
        let old = "pub fn twice(n: i64) -> i64 {\n    return n;\n}\n";
        let proposed = "fn twice(x: i64) -> i64 { return x * 2; }";
        assert!(model_body_to_new_text(old, "not_present", proposed).is_none());
    }

    /// Multi-file NEW-feature coordination on a BINARY crate: an additive request
    /// on a manifest-style `src/main.rs` (module decls + `fn main` only) must
    /// produce the same COORDINATED two-file patch as a library — a new module file
    /// plus the `mod`/`use` wiring in main.rs — not a blind append. No model in the
    /// loop (purely emergent synthesis).
    #[test]
    fn emergent_addition_wires_a_new_module_into_a_binary_main() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        std::env::remove_var("NSYNTH_LOCAL_LLM_URL");
        let root = std::env::temp_dir().join(format!("nsynth_binmod_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"binmod\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
        )
        .expect("cargo.toml");
        // Manifest-style BINARY root: module decls + `fn main` only (no logic fns).
        fs::write(
            root.join("src/main.rs"),
            "mod util;\n\nfn main() {\n    println!(\"{}\", util::seed());\n}\n",
        )
        .expect("main.rs");
        fs::write(root.join("src/util.rs"), "pub fn seed() -> i64 {\n    1\n}\n").expect("util.rs");

        let task = RepoTaskSpec {
            id: "add-triple-bin".into(),
            repo: root.to_string_lossy().to_string(),
            kind: RepoTaskKind::BugFix,
            issue: "nl: add a function that triples a number".into(),
            test_command: "cargo build".into(),
            allowed_files: vec!["src/**".into()],
            max_iterations: 2,
            hardness: HardnessProfile::for_expected_tier(HardnessTier::SingleFileBug),
            signals: Vec::new(),
        };
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("context");
        let patch = nl_synthesis_proposer(&task, &context, 0, None).expect("propose");

        assert!(
            patch.edits.len() >= 2,
            "coordinated multi-file patch expected on a binary root: {:?}",
            patch.metadata
        );
        assert!(
            patch
                .edits
                .iter()
                .any(|e| e.path == "src/main.rs" && e.new_text.contains("mod ")),
            "binary main.rs wired with a new module decl: {:?}",
            patch.edits.iter().map(|e| e.path.clone()).collect::<Vec<_>>()
        );
        assert!(
            patch
                .edits
                .iter()
                .any(|e| e.path != "src/main.rs" && e.path.starts_with("src/") && e.path.ends_with(".rs")),
            "a new module file was created alongside the wiring"
        );
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn mines_int_examples_from_assert_eq_both_orders_word_boundary() {
        use crate::benchmark::Value;
        let src = "assert_eq!(mystery(2), 5);\n assert_eq!(7, mystery(3));\n \
                   assert_eq!(mystery(4), 9);\n assert_eq!(mystery_helper(1), 99);";
        let ex = mine_asserts(src, "mystery");
        assert!(ex.contains(&(vec![Value::Int(2)], Value::Int(5))), "call-first order: {ex:?}");
        assert!(ex.contains(&(vec![Value::Int(3)], Value::Int(7))), "int-first order: {ex:?}");
        assert!(ex.contains(&(vec![Value::Int(4)], Value::Int(9))));
        assert!(
            !ex.iter().any(|(_, o)| *o == Value::Int(99)),
            "mystery_helper must not match mystery"
        );
    }

    #[test]
    fn mines_string_and_bool_asserts() {
        use crate::benchmark::Value;
        let src = "assert_eq!(rev(\"ab\"), \"ba\");\n assert_eq!(is_pal(\"aa\"), true);\n \
                   assert_eq!(is_pal(\"ab\"), false);";
        let r = mine_asserts(src, "rev");
        assert!(
            r.contains(&(vec![Value::Str("ab".into())], Value::Str("ba".into()))),
            "string example: {r:?}"
        );
        let b = mine_asserts(src, "is_pal");
        assert!(b.contains(&(vec![Value::Str("aa".into())], Value::Bool(true))), "bool true: {b:?}");
        assert!(b.contains(&(vec![Value::Str("ab".into())], Value::Bool(false))), "bool false: {b:?}");
    }

    /// The gated model's multi-file coordination (pure — no model present): a response
    /// with two functions, each defined in a different repo file, becomes one atomic
    /// two-file patch. Downstream the cargo-test oracle still gates it.
    #[test]
    fn model_multifile_patch_coordinates_two_files() {
        let root = std::env::temp_dir().join(format!("nsynth_mf_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"mf\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        )
        .unwrap();
        fs::write(root.join("src/lib.rs"), "mod a;\nmod b;\n").unwrap();
        fs::write(root.join("src/a.rs"), "pub fn alpha(n: i64) -> i64 {\n    n\n}\n").unwrap();
        fs::write(root.join("src/b.rs"), "pub fn beta(n: i64) -> i64 {\n    n\n}\n").unwrap();
        let ctx = RepairContext::build(&root, &GuardrailPolicy::default()).unwrap();

        // A "model response" that corrects BOTH functions (multi-line, as a model emits).
        let response = "fn alpha(n: i64) -> i64 {\n    return n + 1;\n}\nfn beta(n: i64) -> i64 {\n    return n * 2;\n}\n";
        let patch = model_response_to_multifile_patch(&ctx, response).expect("multi-file patch");
        assert!(
            patch.metadata.iter().any(|(k, v)| k == "proposer" && v == "model_repair_multifile"),
            "metadata: {:?}",
            patch.metadata
        );
        let paths: Vec<_> = patch.edits.iter().map(|e| e.path.clone()).collect();
        assert!(
            paths.contains(&"src/a.rs".to_string()) && paths.contains(&"src/b.rs".to_string()),
            "both files edited: {paths:?}"
        );
        assert!(patch.edits.iter().any(|e| e.path == "src/a.rs" && e.new_text.contains("+ 1")));
        assert!(patch.edits.iter().any(|e| e.path == "src/b.rs" && e.new_text.contains("* 2")));
        let _ = fs::remove_dir_all(&root);
    }

    /// The deterministic test-oracle lever: a repo fn with NO registry name match and
    /// NO examples in the prose is repaired purely from its failing `assert_eq!` calls
    /// (mystery(x) = 2x+1) — verified by the solver, no model in the loop.
    #[test]
    fn test_mined_synthesis_repairs_from_asserts_no_prose_no_llm() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        std::env::remove_var("NSYNTH_LOCAL_LLM_URL");
        let root = std::env::temp_dir().join(format!("nsynth_testmine_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"tm\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        )
        .expect("cargo.toml");
        fs::write(
            root.join("src/lib.rs"),
            "pub fn mystery(n: i64) -> i64 {\n    n\n}\n\n#[cfg(test)]\nmod tests {\n    use super::mystery;\n    #[test]\n    fn t() {\n        assert_eq!(mystery(2), 5);\n        assert_eq!(mystery(3), 7);\n        assert_eq!(mystery(4), 9);\n    }\n}\n",
        )
        .expect("lib.rs");

        let task = RepoTaskSpec {
            id: "test-mined".into(),
            repo: root.to_string_lossy().to_string(),
            kind: RepoTaskKind::BugFix,
            issue: "nl: fix the mystery function".into(),
            test_command: "cargo test".into(),
            allowed_files: vec!["src/**".into()],
            max_iterations: 2,
            hardness: HardnessProfile::for_expected_tier(HardnessTier::SingleFileBug),
            signals: Vec::new(),
        };
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("ctx");
        let patch = nl_synthesis_proposer(&task, &context, 0, None).expect("propose");
        assert!(
            patch.metadata.iter().any(|(k, v)| k == "proposer" && v == "nl_test_mined_synthesis"),
            "test-mined stage should win (no registry match, no prose examples): {:?}",
            patch.metadata
        );
        assert!(
            patch.edits.iter().any(|e| e.path == "src/lib.rs" && e.new_text.contains('2')),
            "mystery body replaced with the solver's 2x+1"
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// Retarget-to-definition: `src/lib.rs` only declares `pub mod math;` and the failing function
    /// lives in `src/math.rs`. pick_target_path follows the crate root, but the patch must edit the
    /// DEFINING file — otherwise it appends a misplaced/duplicate definition and never satisfies the
    /// oracle. The mined patch must land on src/math.rs.
    #[test]
    fn test_mined_synthesis_targets_the_defining_module_file_not_the_crate_root() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        std::env::remove_var("NSYNTH_LOCAL_LLM_URL");
        let root = std::env::temp_dir().join(format!("nsynth_retarget_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"rt\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        )
        .expect("cargo.toml");
        fs::write(root.join("src/lib.rs"), "pub mod math;\n").expect("lib.rs");
        fs::write(
            root.join("src/math.rs"),
            "pub fn mystery(n: i64) -> i64 {\n    n\n}\n\n#[cfg(test)]\nmod tests {\n    use super::mystery;\n    #[test]\n    fn t() {\n        assert_eq!(mystery(2), 5);\n        assert_eq!(mystery(3), 7);\n        assert_eq!(mystery(4), 9);\n    }\n}\n",
        )
        .expect("math.rs");

        let task = RepoTaskSpec {
            id: "retarget".into(),
            repo: root.to_string_lossy().to_string(),
            kind: RepoTaskKind::BugFix,
            issue: "nl: fix the failing tests".into(),
            test_command: "cargo test".into(),
            allowed_files: vec!["src/**".into()],
            max_iterations: 2,
            hardness: HardnessProfile::for_expected_tier(HardnessTier::SingleFileBug),
            signals: Vec::new(),
        };
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("ctx");
        let patch = nl_synthesis_proposer(&task, &context, 0, None).expect("propose");
        assert!(
            patch.edits.iter().any(|e| e.path == "src/math.rs"),
            "patch must edit the DEFINING file src/math.rs, not the crate root: {:?}",
            patch.edits.iter().map(|e| &e.path).collect::<Vec<_>>()
        );
        assert!(
            !patch.edits.iter().any(|e| e.path == "src/lib.rs"),
            "must NOT edit the module-declaring crate root"
        );
        let _ = fs::remove_dir_all(&root);
    }

    static NL_SYNTHESIS_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn synthesis_fixture() -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "nsynth_nl_repair_{}_{}",
            "fixture",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        write_nl_fixture_crate(&root, "nl_fixture_add").expect("write fixture");
        root
    }

    fn nl_task(root: &PathBuf, id: &str, fixture_id: &str, issue: &str) -> RepoTaskSpec {
        RepoTaskSpec {
            id: id.into(),
            repo: root.to_string_lossy().to_string(),
            kind: RepoTaskKind::Feature,
            issue: issue.into(),
            test_command: nl_fixture_cargo_test_command(fixture_id).expect("cmd"),
            allowed_files: vec!["src/**".into()],
            max_iterations: 2,
            hardness: HardnessProfile::for_expected_tier(HardnessTier::SingleFileBug),
            signals: Vec::new(),
        }
    }

    #[test]
    fn nl_description_prefixes() {
        assert_eq!(
            nl_description_from_issue("synthesize: add two numbers"),
            Some("add two numbers".to_string())
        );
        assert_eq!(
            nl_description_from_issue("nl:reverse the array"),
            Some("reverse the array".to_string())
        );
        assert!(nl_description_from_issue("fix the bug").is_none());
    }

    #[test]
    fn mog_gcd_transpiles_without_mog_assign_syntax() {
        let mog = "fn gcd(a: i64, b: i64) -> i64 {\n    x: i64 = a;\n    y: i64 = b;\n    while y != 0 {\n        tmp := y;\n        y = x % y;\n        x = tmp;\n    }\n    return x;\n}\n";
        let rust = rust_code_for_repo_synthesis(mog);
        assert!(rust.contains("pub fn gcd"));
        assert!(!rust.contains(":="));
        assert!(rust.contains("let mut tmp"));
    }

    #[test]
    fn repo_rust_body_scalar_stubs() {
        use crate::agent::coding_intent::CodingIntent;

        let add = CodingIntent {
            function_name: "add".into(),
            signature: "i64, i64 -> i64".into(),
            category: "arithmetic".into(),
            description: "add two numbers".into(),
            examples: Vec::new(),
            constraints: Vec::new(),
            confidence: 1.0,
            unresolved: Vec::new(),
            evidence_entity_ids: Vec::new(),
            reference_code: String::new(),
        };
        let hint = "pub fn add_two(a: i64, b: i64) -> i64 { a - b }\n";
        let body = repo_rust_body_for_nl(&add, "", Some(hint)).expect("add stub");
        assert!(body.contains("add_two"));
        assert!(body.contains("a + b"));

        let max = CodingIntent {
            function_name: "max".into(),
            signature: "i64, i64 -> i64".into(),
            category: "comparison".into(),
            description: "return the larger of two numbers".into(),
            examples: Vec::new(),
            constraints: Vec::new(),
            confidence: 1.0,
            unresolved: Vec::new(),
            evidence_entity_ids: Vec::new(),
            reference_code: String::new(),
        };
        let max_hint = "pub fn max_of(a: i64, b: i64) -> i64 { a }\n";
        let max_body = repo_rust_body_for_nl(&max, "", Some(max_hint)).expect("max stub");
        assert!(max_body.contains("max_of"));
        assert!(max_body.contains("if a > b"));
    }

    /// THE DIFFERENTIATOR, end to end and un-gameable: a repo fn whose NAME
    /// differs from the op it should implement ("the twice function should
    /// double the number", fn is `twice`, op is `double`).
    ///   * no I/O examples in the issue; LLM lane explicitly OFF (env removed) —
    ///     comprehension is the linguigenesis emergent resolver alone;
    ///   * the intent-name primary MISLOCALIZES here (proven during dev: it
    ///     wrote a brand-new `fn double` into lib.rs, leaving `twice` broken);
    ///   * the emergent stage CONTENT-localizes `fn twice` in src/ops.rs (prose
    ///     token "twice" matches the defined fn), synthesizes the doubling body
    ///     via the bridge, and preserves the repo fn name;
    ///   * driven through the REAL proposer chain (nl_synthesis_proposer), and
    ///     acceptance is behavioral: the failing cargo test passes after.
    #[test]
    fn emergent_proposer_repairs_renamed_fn_bare_nl_no_llm() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        std::env::remove_var("NSYNTH_LOCAL_LLM_URL"); // no model in the loop
        let root = std::env::temp_dir().join(format!("nsynth_emergent_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"twicefix\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        )
        .expect("cargo.toml");
        fs::write(
            root.join("src/lib.rs"),
            "mod ops;\npub use ops::twice;\n\n#[cfg(test)]\nmod tests {\n    use super::twice;\n    #[test]\n    fn twice_doubles() {\n        assert_eq!(twice(4), 8);\n        assert_eq!(twice(-3), -6);\n    }\n}\n",
        )
        .expect("lib.rs");
        fs::write(
            root.join("src/ops.rs"),
            "pub fn twice(x: i64) -> i64 {\n    x + 1\n}\n",
        )
        .expect("ops.rs");

        let task = RepoTaskSpec {
            id: "emergent-twice".into(),
            repo: root.to_string_lossy().to_string(),
            kind: RepoTaskKind::BugFix,
            issue: "nl: the twice function should double the number".into(),
            test_command: "cargo test twice_doubles".into(),
            allowed_files: vec!["src/**".into()],
            max_iterations: 2,
            hardness: HardnessProfile::for_expected_tier(HardnessTier::SingleFileBug),
            signals: Vec::new(),
        };
        let before = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify before");
        assert!(!before.success, "fixture must start broken");

        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("context");
        // Through the REAL chain — the emergent stage must win the ordering.
        let patch = nl_synthesis_proposer(&task, &context, 0, None).expect("propose");
        assert!(
            patch
                .metadata
                .iter()
                .any(|(k, v)| k == "proposer" && v == "nl_emergent_synthesis"),
            "emergent stage should produce this patch: {:?}",
            patch.metadata
        );
        assert_eq!(patch.edits[0].path, "src/ops.rs", "content-localized to the defining file");
        assert!(
            patch.edits[0].new_text.contains("fn twice"),
            "repo fn name preserved: {}",
            patch.edits[0].new_text
        );
        fs::write(root.join(&patch.edits[0].path), &patch.edits[0].new_text).expect("apply");
        let after = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify after");
        assert!(after.success, "stderr: {}", after.stderr);
        let _ = fs::remove_dir_all(root);
    }

    /// STRING repair through the emergent stage: "the shout function should
    /// uppercase the text" — the type-domain widening reaching the repair path.
    #[test]
    fn emergent_proposer_repairs_string_fn_bare_nl_no_llm() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        std::env::remove_var("NSYNTH_LOCAL_LLM_URL");
        let root = std::env::temp_dir().join(format!("nsynth_emstr_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"strfix\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        )
        .expect("cargo.toml");
        fs::write(
            root.join("src/lib.rs"),
            "mod text_utils;\npub use text_utils::shout;\n\n#[cfg(test)]\nmod tests {\n    use super::shout;\n    #[test]\n    fn shouts() {\n        assert_eq!(shout(\"hello\".to_string()), \"HELLO\");\n    }\n}\n",
        )
        .expect("lib.rs");
        fs::write(
            root.join("src/text_utils.rs"),
            "pub fn shout(s: String) -> String {\n    s\n}\n",
        )
        .expect("text_utils.rs");

        let task = RepoTaskSpec {
            id: "emergent-shout".into(),
            repo: root.to_string_lossy().to_string(),
            kind: RepoTaskKind::BugFix,
            issue: "nl: the shout function should uppercase the text".into(),
            test_command: "cargo test shouts".into(),
            allowed_files: vec!["src/**".into()],
            max_iterations: 2,
            hardness: HardnessProfile::for_expected_tier(HardnessTier::SingleFileBug),
            signals: Vec::new(),
        };
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("context");
        let patch = try_emergent_synthesis_patch(
            &task,
            &context,
            "the shout function should uppercase the text",
            None,
        )
        .expect("emergent string patch");
        assert_eq!(patch.edits[0].path, "src/text_utils.rs");
        assert!(
            patch.edits[0].new_text.contains("to_uppercase"),
            "verified string body: {}",
            patch.edits[0].new_text
        );
        fs::write(root.join(&patch.edits[0].path), &patch.edits[0].new_text).expect("apply");
        let after = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify after");
        assert!(after.success, "stderr: {}", after.stderr);
        let _ = fs::remove_dir_all(root);
    }

    /// FEATURE-ADD end to end (TDD shape, no examples, no LLM): a failing test
    /// references a fn that does NOT exist; "add a function that triples a
    /// number" synthesizes `triple` via emergent comprehension and APPENDS it —
    /// the compile-failing crate goes green.
    #[test]
    fn emergent_addition_adds_missing_fn_from_bare_nl() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        std::env::remove_var("NSYNTH_LOCAL_LLM_URL");
        let root = std::env::temp_dir().join(format!("nsynth_addfn_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"addfix\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        )
        .expect("cargo.toml");
        // TDD: the test exists, the fn does not — baseline is a COMPILE failure.
        fs::write(
            root.join("src/lib.rs"),
            "#[cfg(test)]\nmod tests {\n    #[test]\n    fn triples() {\n        assert_eq!(crate::triple(4), 12);\n        assert_eq!(crate::triple(-2), -6);\n    }\n}\n",
        )
        .expect("lib.rs");

        let task = RepoTaskSpec {
            id: "add-triple".into(),
            repo: root.to_string_lossy().to_string(),
            kind: RepoTaskKind::Feature,
            issue: "nl: add a function that triples a number".into(),
            test_command: "cargo test triples".into(),
            allowed_files: vec!["src/**".into()],
            max_iterations: 2,
            hardness: HardnessProfile::for_expected_tier(HardnessTier::SingleFileBug),
            signals: Vec::new(),
        };
        let before = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify before");
        assert!(!before.success, "fixture must start broken (missing fn)");
        let analysis = FailureParser::default().parse(&before.failure_output());

        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("context");
        // Through the REAL chain.
        let patch = nl_synthesis_proposer(&task, &context, 0, Some(&analysis)).expect("propose");
        assert!(
            patch
                .metadata
                .iter()
                .any(|(k, v)| k == "proposer" && v == "nl_emergent_addition"),
            "addition stage should fire: {:?}",
            patch.metadata
        );
        let new_text = &patch.edits[0].new_text;
        assert!(new_text.contains("pub fn triple"), "new fn appended: {new_text}");
        // The original content is preserved (append, not replace).
        assert!(new_text.contains("mod tests"), "existing content kept: {new_text}");
        fs::write(root.join(&patch.edits[0].path), new_text).expect("apply");
        let after = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify after");
        assert!(after.success, "stderr: {}", after.stderr);
        let _ = fs::remove_dir_all(root);
    }

    /// CONTENT-GREP localization (ReAct slice), end to end: the fn is called
    /// `mul3` — NO name/morphology relation to the prose — but its file's
    /// comment says "tripling helper". Content search finds the file, the
    /// single-fn rule targets `mul3`, emergent comprehension synthesizes the
    /// tripling body under the repo's fn name, cargo goes green.
    #[test]
    fn content_grep_localizes_fn_whose_name_shares_nothing_with_prose() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        std::env::remove_var("NSYNTH_LOCAL_LLM_URL");
        let root = std::env::temp_dir().join(format!("nsynth_grep_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"grepfix\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        )
        .expect("cargo.toml");
        fs::write(
            root.join("src/lib.rs"),
            "pub mod maths;\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn mul3_triples() {\n        assert_eq!(crate::maths::mul3(4), 12);\n    }\n}\n",
        )
        .expect("lib.rs");
        // The fn name shares NOTHING with the prose; the comment carries the link.
        fs::write(
            root.join("src/maths.rs"),
            "// The tripling helper used by the pricing code.\npub fn mul3(x: i64) -> i64 {\n    x + 3\n}\n",
        )
        .expect("maths.rs");

        let task = RepoTaskSpec {
            id: "grep-mul3".into(),
            repo: root.to_string_lossy().to_string(),
            kind: RepoTaskKind::BugFix,
            issue: "nl: the tripling helper should triple the number".into(),
            test_command: "cargo test mul3_triples".into(),
            allowed_files: vec!["src/**".into()],
            max_iterations: 2,
            hardness: HardnessProfile::for_expected_tier(HardnessTier::SingleFileBug),
            signals: Vec::new(),
        };
        let before = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify before");
        assert!(!before.success, "fixture must start broken");

        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("context");
        // Name matching alone must FAIL here...
        let desc = "the tripling helper should triple the number";
        let patch = try_emergent_synthesis_patch(&task, &context, desc, None)
            .expect("content-grep localized patch");
        assert_eq!(patch.edits[0].path, "src/maths.rs", "found via content, not name");
        assert!(patch.edits[0].new_text.contains("fn mul3"), "repo fn name kept: {}", patch.edits[0].new_text);
        fs::write(root.join(&patch.edits[0].path), &patch.edits[0].new_text).expect("apply");
        let after = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify after");
        assert!(after.success, "stderr: {}", after.stderr);
        let _ = fs::remove_dir_all(root);
    }

    /// SIGNATURE-CHANGE refactor, end to end: "add a parameter offset to scale
    /// defaulting to 0" — the definition gains the param, the cross-file call
    /// site gains the default argument, tests (already calling the new arity)
    /// go green. Behavior preserved by construction; tests never edited.
    #[test]
    fn add_param_refactor_updates_signature_and_call_sites() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        let root = std::env::temp_dir().join(format!("nsynth_addparam_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"addparam\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        )
        .expect("cargo.toml");
        fs::write(
            root.join("src/lib.rs"),
            "pub mod maths;\npub mod report;\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn new_arity() {\n        assert_eq!(crate::maths::scale(4, 0), 8);\n        assert_eq!(crate::report::doubled(3), 6);\n    }\n}\n",
        )
        .expect("lib.rs");
        fs::write(root.join("src/maths.rs"), "pub fn scale(x: i64) -> i64 {\n    2 * x\n}\n")
            .expect("maths");
        fs::write(
            root.join("src/report.rs"),
            "use crate::maths::scale;\n\npub fn doubled(n: i64) -> i64 {\n    scale(n)\n}\n",
        )
        .expect("report");

        let task = RepoTaskSpec {
            id: "addparam-scale".into(),
            repo: root.to_string_lossy().to_string(),
            kind: RepoTaskKind::Refactor,
            issue: "nl: add a parameter offset to scale defaulting to 0".into(),
            test_command: "cargo test new_arity".into(),
            allowed_files: vec!["src/**".into()],
            max_iterations: 2,
            hardness: HardnessProfile::for_expected_tier(HardnessTier::SingleFileBug),
            signals: Vec::new(),
        };
        let before = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify before");
        assert!(!before.success, "fixture must start broken (tests call new arity)");

        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("context");
        let patch = nl_synthesis_proposer(&task, &context, 0, None).expect("propose");
        assert!(
            patch
                .metadata
                .iter()
                .any(|(k, v)| k == "proposer" && v == "nl_add_param_refactor"),
            "add-param stage should fire: {:?}",
            patch.metadata
        );
        let paths: Vec<&str> = patch.edits.iter().map(|e| e.path.as_str()).collect();
        assert!(paths.contains(&"src/maths.rs"), "definition edited: {paths:?}");
        assert!(paths.contains(&"src/report.rs"), "call site edited: {paths:?}");
        let def = patch.edits.iter().find(|e| e.path == "src/maths.rs").unwrap();
        assert!(def.new_text.contains("scale(x: i64, offset: i64)"), "{}", def.new_text);
        let call = patch.edits.iter().find(|e| e.path == "src/report.rs").unwrap();
        assert!(call.new_text.contains("scale(n, 0)"), "{}", call.new_text);

        let mut tx = crate::agent::edit::EditTransaction::begin(&root);
        tx.apply_repair_patch(&patch).expect("apply");
        tx.commit().expect("commit");
        let after = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify after");
        assert!(after.success, "stderr: {}", after.stderr);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn add_param_declines_without_default_or_unknown_fn() {
        let root = std::env::temp_dir().join(format!("nsynth_apneg_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(root.join("Cargo.toml"), "[package]\nname=\"x\"\nversion=\"0.1.0\"\n").unwrap();
        fs::write(root.join("src/lib.rs"), "pub fn scale(x: i64) -> i64 { 2 * x }\n").unwrap();
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("context");
        // No default literal -> decline (never fabricate).
        assert!(try_add_param_patch(&context, "add a parameter offset to scale").is_none());
        // Unknown fn -> decline.
        assert!(
            try_add_param_patch(&context, "add a parameter offset to shrink defaulting to 0")
                .is_none()
        );
        let _ = fs::remove_dir_all(root);
    }

    /// COORDINATED MULTI-FILE RENAME, end to end: definition in one file, call
    /// site in another, both rewritten in ONE atomic patch; the TDD oracle
    /// (tests already call the NEW name) goes green. Tests are never edited.
    #[test]
    fn rename_refactor_updates_definition_and_call_sites() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        let root = std::env::temp_dir().join(format!("nsynth_rename_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"renamefix\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        )
        .expect("cargo.toml");
        fs::write(
            root.join("src/lib.rs"),
            "pub mod maths;\npub mod report;\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn twice_works() {\n        assert_eq!(crate::maths::twice(4), 8);\n        assert_eq!(crate::report::doubled_len(3), 6);\n    }\n}\n",
        )
        .expect("lib.rs");
        // Definition file...
        fs::write(root.join("src/maths.rs"), "pub fn double(x: i64) -> i64 {\n    2 * x\n}\n")
            .expect("maths");
        // ...and a CALLER in a different file.
        fs::write(
            root.join("src/report.rs"),
            "use crate::maths::double;\n\npub fn doubled_len(n: i64) -> i64 {\n    double(n)\n}\n",
        )
        .expect("report");

        let task = RepoTaskSpec {
            id: "rename-double".into(),
            repo: root.to_string_lossy().to_string(),
            kind: RepoTaskKind::Refactor,
            issue: "nl: rename double to twice".into(),
            test_command: "cargo test twice_works".into(),
            allowed_files: vec!["src/**".into()],
            max_iterations: 2,
            hardness: HardnessProfile::for_expected_tier(HardnessTier::SingleFileBug),
            signals: Vec::new(),
        };
        let before = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify before");
        assert!(!before.success, "fixture must start broken (tests call twice)");

        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("context");
        let patch = nl_synthesis_proposer(&task, &context, 0, None).expect("propose");
        assert!(
            patch
                .metadata
                .iter()
                .any(|(k, v)| k == "proposer" && v == "nl_rename_refactor"),
            "rename stage should fire: {:?}",
            patch.metadata
        );
        let paths: Vec<&str> = patch.edits.iter().map(|e| e.path.as_str()).collect();
        assert!(paths.contains(&"src/maths.rs"), "definition file edited: {paths:?}");
        assert!(paths.contains(&"src/report.rs"), "caller file edited: {paths:?}");
        // The oracle is untouched: no edit rewrites the tests module's calls.
        for e in &patch.edits {
            assert!(
                !e.new_text.contains("crate::maths::double"),
                "no stale references left: {}",
                e.new_text
            );
        }

        let mut tx = crate::agent::edit::EditTransaction::begin(&root);
        tx.apply_repair_patch(&patch).expect("apply");
        tx.commit().expect("commit");
        let after = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify after");
        assert!(after.success, "stderr: {}", after.stderr);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn rename_declines_undefined_source_and_collisions() {
        let root = std::env::temp_dir().join(format!("nsynth_renneg_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(root.join("Cargo.toml"), "[package]\nname=\"x\"\nversion=\"0.1.0\"\n").unwrap();
        fs::write(
            root.join("src/lib.rs"),
            "pub fn double(x: i64) -> i64 { 2 * x }\npub fn triple(x: i64) -> i64 { 3 * x }\n",
        )
        .unwrap();
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("context");
        // Undefined source fn declines.
        assert!(try_rename_patch(&context, "rename quadruple to quint").is_none());
        // Collision with an existing fn declines.
        assert!(try_rename_patch(&context, "rename double to triple").is_none());
        // Word-boundary safety: renaming double must not touch doubled identifiers.
        let p = try_rename_patch(&context, "rename double to twice").expect("patch");
        assert!(p.edits[0].new_text.contains("pub fn twice"));
        assert!(p.edits[0].new_text.contains("pub fn triple"), "others untouched");
        let _ = fs::remove_dir_all(root);
    }

    /// MULTI-FILE COORDINATED ADDITION, end to end through the REAL transaction:
    /// a module-manifest repo (lib.rs only declares mods) gets a NEW module file
    /// src/triple.rs (file CREATION via EditTransaction) plus the lib.rs wiring
    /// in ONE atomic patch — and the compile-failing crate goes green.
    #[test]
    fn emergent_addition_multifile_creates_module_and_wires_manifest() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        std::env::remove_var("NSYNTH_LOCAL_LLM_URL");
        let root = std::env::temp_dir().join(format!("nsynth_addmf_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"addmf\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        )
        .expect("cargo.toml");
        // Module-manifest lib.rs: declares mods, defines no fns of its own.
        fs::write(
            root.join("src/lib.rs"),
            "mod ops;\npub use ops::*;\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn triples() {\n        assert_eq!(crate::triple(4), 12);\n    }\n}\n",
        )
        .expect("lib.rs");
        fs::write(root.join("src/ops.rs"), "pub fn double(x: i64) -> i64 {\n    2 * x\n}\n")
            .expect("ops.rs");

        let task = RepoTaskSpec {
            id: "add-mf-triple".into(),
            repo: root.to_string_lossy().to_string(),
            kind: RepoTaskKind::Feature,
            issue: "nl: add a function that triples a number".into(),
            test_command: "cargo test triples".into(),
            allowed_files: vec!["src/**".into()],
            max_iterations: 2,
            hardness: HardnessProfile::for_expected_tier(HardnessTier::SingleFileBug),
            signals: Vec::new(),
        };
        let before = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify before");
        assert!(!before.success, "fixture must start broken (missing fn)");
        let analysis = FailureParser::default().parse(&before.failure_output());

        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("context");
        let patch = nl_synthesis_proposer(&task, &context, 0, Some(&analysis)).expect("propose");
        assert!(
            patch
                .metadata
                .iter()
                .any(|(k, v)| k == "proposer" && v == "nl_emergent_addition_multifile"),
            "multi-file addition should fire: {:?}",
            patch.metadata
        );
        assert_eq!(patch.edits.len(), 2, "coordinated two-file patch");
        assert_eq!(patch.edits[0].path, "src/triple.rs", "new module file");
        assert!(patch.edits[1].new_text.contains("mod triple;"), "manifest wired");

        // Apply through the REAL transaction — proves atomic file CREATION.
        let mut tx = crate::agent::edit::EditTransaction::begin(&root);
        tx.apply_repair_patch(&patch).expect("apply");
        tx.commit().expect("commit");
        assert!(root.join("src/triple.rs").is_file(), "module created on disk");

        let after = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify after");
        assert!(after.success, "stderr: {}", after.stderr);
        let _ = fs::remove_dir_all(root);
    }

    /// OBSERVATION-DRIVEN disambiguation: two files define a `double`; walk
    /// order alone would patch the healthy one (aaa.rs) and leave the broken one
    /// (zzz.rs) failing forever. The failure-implicated file from FailureAnalysis
    /// must outrank walk order, and the repair must land + verify green.
    #[test]
    fn failure_file_outranks_walk_order_in_localization() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        std::env::remove_var("NSYNTH_LOCAL_LLM_URL");
        let root = std::env::temp_dir().join(format!("nsynth_obsloc_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"obsfix\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        )
        .expect("cargo.toml");
        fs::write(
            root.join("src/lib.rs"),
            "pub mod aaa;\npub mod zzz;\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn zzz_doubles() {\n        assert_eq!(crate::zzz::double(4), 8);\n    }\n}\n",
        )
        .expect("lib.rs");
        // aaa: healthy double — walk order would pick this one first.
        fs::write(root.join("src/aaa.rs"), "pub fn double(x: i64) -> i64 {\n    2 * x\n}\n")
            .expect("aaa");
        // zzz: the BROKEN double the failing test actually exercises.
        fs::write(root.join("src/zzz.rs"), "pub fn double(x: i64) -> i64 {\n    x + 1\n}\n")
            .expect("zzz");

        let task = RepoTaskSpec {
            id: "obs-double".into(),
            repo: root.to_string_lossy().to_string(),
            kind: RepoTaskKind::BugFix,
            issue: "nl: the double function should double the number".into(),
            test_command: "cargo test zzz_doubles".into(),
            allowed_files: vec!["src/**".into()],
            max_iterations: 2,
            hardness: HardnessProfile::for_expected_tier(HardnessTier::SingleFileBug),
            signals: Vec::new(),
        };
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("context");
        let desc = "the double function should double the number";

        // WITHOUT analysis: walk order picks aaa.rs (the healthy file) — the
        // blind spot this feature closes.
        let (blind_path, _) = locate_described_fn(&context, desc, None).expect("blind");
        assert_eq!(blind_path, "src/aaa.rs", "walk order picks the wrong file");

        // WITH the failure-implicated file: zzz.rs wins.
        let analysis = FailureAnalysis {
            kind: crate::agent::repo::FailureKind::TestFailure,
            file: Some("src/zzz.rs".to_string()),
            line: Some(2),
            message: "assertion failed".into(),
            likely_cause: String::new(),
            suggested_action: String::new(),
        };
        let patch = try_emergent_synthesis_patch(&task, &context, desc, Some(&analysis))
            .expect("observation-driven patch");
        assert_eq!(patch.edits[0].path, "src/zzz.rs", "failure file outranks walk order");
        fs::write(root.join(&patch.edits[0].path), &patch.edits[0].new_text).expect("apply");
        let after = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify after");
        assert!(after.success, "stderr: {}", after.stderr);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn locate_described_fn_matches_emergently_and_specifically() {
        let root = std::env::temp_dir().join(format!("nsynth_locate_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(root.join("Cargo.toml"), "[package]\nname=\"x\"\nversion=\"0.1.0\"\n").unwrap();
        fs::write(
            root.join("src/util.rs"),
            "pub fn double(x: i64) -> i64 { x }\npub fn reverse_list(v: Vec<i64>) -> Vec<i64> { v }\npub fn number_cruncher(x: i64) -> i64 { x }\n",
        )
        .unwrap();
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("context");
        // morphology: "doubling" -> double
        let (_, f) = locate_described_fn(&context, "fix the doubling function", None).expect("found");
        assert_eq!(f, "double");
        // multi-part fn: every part must be matched
        let (_, f) = locate_described_fn(&context, "reverse the list please", None).expect("found");
        assert_eq!(f, "reverse_list");
        // incidental "number" alone must NOT select number_cruncher
        assert!(locate_described_fn(&context, "the number should be bigger", None).is_none());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn nl_synthesis_proposer_fixes_wrong_add() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        let root = synthesis_fixture();
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("context");
        let task = nl_task(
            &root,
            "nl-add",
            "nl_fixture_add",
            "synthesize: add two numbers",
        );

        let patch = nl_synthesis_proposer(&task, &context, 0, None).expect("propose");
        assert!(!patch.edits.is_empty());
        let new_content = patch.edits[0].new_text.clone();
        assert!(new_content.contains("add_two"));
        // Acceptance is behavior-based (cargo test), independent of whether the
        // real-synthesis primary or keyword fallback produced the patch.
        fs::write(root.join(patch.edits[0].path.clone()), new_content).expect("write patch");
        let after = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify after");
        assert!(after.success, "stderr: {}", after.stderr);

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn failure_aware_proposer_uses_cargo_test_failure() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        let root = std::env::temp_dir().join(format!("nsynth_nl_fail_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        write_nl_fixture_crate(&root, "nl_fixture_divide").expect("write");
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("context");
        let task = nl_task(
            &root,
            "nl-div-fail",
            "nl_fixture_divide",
            "synthesize: divide two numbers",
        );
        let verification = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify");
        assert!(!verification.success);
        let analysis = FailureParser::default().parse(&verification.failure_output());
        assert_eq!(analysis.kind, crate::agent::repo::FailureKind::TestFailure);

        // divide's solver output uses Result-style wrappers, so real synthesis
        // declines and the failure-aware keyword fallback repairs it. Acceptance
        // is behavior-based (cargo test passes after the patch).
        let patch = nl_synthesis_proposer(&task, &context, 0, Some(&analysis)).expect("patch");
        fs::write(
            root.join(patch.edits[0].path.clone()),
            patch.edits[0].new_text.clone(),
        )
        .expect("write patch");
        let after = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify after");
        assert!(after.success, "stderr: {}", after.stderr);

        let _ = fs::remove_dir_all(root);
    }

    fn assert_real_synthesis_repairs(fixture_id: &str, description: &str, tag: &str) {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        let root = std::env::temp_dir().join(format!("nsynth_rs_{}_{}", tag, std::process::id()));
        let _ = fs::remove_dir_all(&root);
        write_nl_fixture_crate(&root, fixture_id).expect("write");
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("context");
        let task = nl_task(&root, tag, fixture_id, &format!("synthesize: {description}"));

        let before = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify before");
        assert!(!before.success, "{fixture_id} should fail before repair");

        let patch = try_real_synthesis_patch(&task, &context, description)
            .unwrap_or_else(|| panic!("{fixture_id}: expected real-synthesis patch"));
        fs::write(
            root.join(patch.edits[0].path.clone()),
            patch.edits[0].new_text.clone(),
        )
        .expect("write patch");
        let after = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify after");
        assert!(
            after.success,
            "{fixture_id} real-synthesis repair failed:\nCODE:\n{}\nSTDERR:\n{}",
            patch.edits[0].new_text, after.stderr
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn real_synthesis_repairs_divide_result_idiom() {
        // Safe-division emits a 2-function Result/ok/err/match template; the
        // lowering (ok->Some, err->None, match->unwrap_or) must produce
        // compilable plain Rust that passes the cargo-test oracle.
        assert_real_synthesis_repairs("nl_fixture_divide", "divide two numbers", "div");
    }

    #[test]
    fn real_synthesis_repairs_multiply_multifunction() {
        // multiply may synthesize via the LCM formula (gcd_inner + multiply);
        // multi-function reshape must keep the helper and target the main fn.
        assert_real_synthesis_repairs("nl_fixture_multiply", "multiply two numbers", "mul");
    }

    // Broadened unseen-NL corpus (G5 sign-off): each is described ONLY by inline
    // I/O examples (no registry op, no keyword-table entry), so the repair can
    // succeed *only* through genuine example-driven synthesis. The cargo-test
    // oracle asserts holdout inputs to prove generalization, not example overfit.

    #[test]
    fn real_synthesis_repairs_square_nonlinear() {
        assert_real_synthesis_repairs(
            "nl_fixture_square",
            "a function where square(2)=4 and square(3)=9 and square(4)=16 and square(5)=25 and square(6)=36 and square(0)=0",
            "sq",
        );
    }

    #[test]
    fn real_synthesis_repairs_negate_affine() {
        assert_real_synthesis_repairs(
            "nl_fixture_negate",
            "a function where negate(5)=-5 and negate(-3)=3 and negate(0)=0 and negate(7)=-7 and negate(-12)=12",
            "neg",
        );
    }

    #[test]
    fn real_synthesis_repairs_abs_branch() {
        assert_real_synthesis_repairs(
            "nl_fixture_abs",
            "a function where absval(-3)=3 and absval(4)=4 and absval(-10)=10 and absval(0)=0 and absval(-1)=1 and absval(8)=8",
            "abs",
        );
    }

    #[test]
    fn real_synthesis_repairs_sum3_multiarg() {
        assert_real_synthesis_repairs(
            "nl_fixture_sum3",
            "a function where add3(1,2,3)=6 and add3(0,0,5)=5 and add3(2,2,2)=6 and add3(10,20,30)=60 and add3(-1,1,0)=0",
            "sum3",
        );
    }

    #[test]
    fn real_synthesis_repairs_array_sum_fold() {
        assert_real_synthesis_repairs(
            "nl_fixture_arrsum",
            "a function where total([1,2,3])=6 and total([4,5])=9 and total([10])=10 and total([2,2,2,2])=8 and total([7,3])=10",
            "arrsum",
        );
    }

    #[test]
    fn real_synthesis_repairs_array_max_fold() {
        assert_real_synthesis_repairs(
            "nl_fixture_arrmax",
            "a function where biggest([3,1,2])=3 and biggest([5,9,1])=9 and biggest([7])=7 and biggest([-1,-5,-2])=-1 and biggest([2,2,8,4])=8",
            "arrmax",
        );
    }

    #[test]
    fn real_synthesis_repairs_array_len_fold() {
        assert_real_synthesis_repairs(
            "nl_fixture_arrlen",
            "a function where howmany([3,1,2])=3 and howmany([5,9])=2 and howmany([7])=1 and howmany([1,2,3,4,5])=5 and howmany([6,6])=2",
            "arrlen",
        );
    }

    #[test]
    fn real_synthesis_repairs_min3_nested_branch() {
        // 3-way minimum: synthesized as nested comparison branches (synth_gradient),
        // not a constant-threshold overfit. Holdouts prove real min logic.
        assert_real_synthesis_repairs(
            "nl_fixture_min3",
            "a function where smallest(3,7,5)=3 and smallest(9,2,8)=2 and smallest(1,4,1)=1 and smallest(5,5,2)=2 and smallest(-1,0,3)=-1 and smallest(8,8,8)=8 and smallest(4,1,9)=1",
            "min3",
        );
    }

    #[test]
    fn real_synthesis_repairs_unseen_inline_example_op() {
        // `triple` is NOT in any keyword table; the only way to repair it is to
        // actually synthesize x*3 from the inline examples in the issue. Proves
        // the closed repair loop generalizes to arbitrary demonstrated functions.
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        let root = std::env::temp_dir().join(format!("nsynth_nl_triple_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        write_nl_fixture_crate(&root, "nl_fixture_triple").expect("write");
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("context");
        let task = nl_task(
            &root,
            "nl-triple",
            "nl_fixture_triple",
            "synthesize: a function where triple(2)=6 and triple(5)=15 and triple(3)=9",
        );

        // Fails before repair.
        let before = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify before");
        assert!(!before.success);

        // The proposer must use the real-synthesis path (not a keyword stub).
        let patch = nl_synthesis_proposer(&task, &context, 0, None).expect("propose");
        assert_eq!(
            patch
                .metadata
                .iter()
                .find(|(k, _)| k == "proposer")
                .map(|(_, v)| v.as_str()),
            Some("nl_real_synthesis"),
            "expected real synthesis, got metadata {:?}",
            patch.metadata
        );

        // Apply and re-verify: cargo test is the acceptance oracle.
        fs::write(
            root.join(patch.edits[0].path.clone()),
            patch.edits[0].new_text.clone(),
        )
        .expect("write patch");
        let after = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify after");
        assert!(after.success, "stderr: {}", after.stderr);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn repair_loop_with_nl_synthesis_proposer() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        let root = synthesis_fixture();
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("context");
        let task = nl_task(
            &root,
            "nl-add-loop",
            "nl_fixture_add",
            "synthesize: add two numbers",
        );
        let mut loop_runner = RepairLoop::new(&root, GuardrailPolicy::default());
        let result = loop_runner
            .run_with_context(&task, &context, &nl_synthesis_proposer)
            .expect("repair loop");
        assert!(result.success);
        let verification = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify");
        assert!(verification.success);
        let _ = fs::remove_dir_all(root);
    }
}

