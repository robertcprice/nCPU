//! NL synthesis proposer for the repair loop (Package B/H bridge).
//!
//! When a `RepoTaskSpec.issue` begins with `nl:` or `synthesize:`, this proposer
//! runs the Linguigenesis-native `AgentRun` path and writes verified synthesis
//! output into an allowed repository file.

use crate::agent::agent_run::AgentRun;
use crate::agent::coding_intent::CodingIntent;
use crate::agent::repo::{
    FailureAnalysis, FailureKind, GuardrailPolicy, RepairContext, RepairEdit, RepairPatch, RepairVerification, RepairVerifier, RepoTaskSpec, RepoTaskKind,
};
use std::collections::HashMap;
use std::fs;
use std::hash::{Hash, Hasher};
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

    // EXTRACT-HELPER refactor (Lever D): "extract the duplicated expression into a helper called
    // X" hoists a repeated pure-i64 sub-expression into `fn X(..) -> i64` and rewrites call sites.
    // Structural like rename/add-param, so it is hoisted above the body-swap synthesis stages; it
    // only fires on an explicit extract instruction for a Refactor task and declines otherwise.
    if matches!(task.kind, RepoTaskKind::Refactor) {
        if let Some(patch) = try_extract_helper_patch(context, &description) {
            return Ok(patch);
        }
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
    if let Some(patch) = try_test_mined_synthesis_patch(task, context, &description, analysis) {
        return Ok(patch);
    }

    if let Some(patch) = try_nl_repo_fast_patch(task, context, &description, analysis) {
        return Ok(patch);
    }

    // GATED MODEL fallback: every deterministic path declined. Two optional,
    // untrusted local-LLM lanes, inert without NSYNTH_LOCAL_LLM_URL. Deterministic-
    // first, model-last. The INTENT lane goes first (model proposes a SPEC, the engine
    // synthesizes + verifies, and an accepted solve DISTILLS so the model teaches
    // once); the direct-body REPAIR lane is the fallback. Both are still gated by the
    // cargo-test oracle downstream.
    if let Some(patch) = try_model_intent_patch(task, context, &description) {
        return Ok(patch);
    }
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
    // GATE against the FAILING TEST's own asserts. This stage grounds via the low-confidence
    // prose bridge and solves the INTENT's canonical examples — which say nothing about THIS
    // repo's failing test. A mis-grounding (count_positives -> reverse-digits, mode -> `last`,
    // prefix-sums -> `array_sum`) otherwise returns a WRONG patch and, because this stage runs
    // first in the ladder, short-circuits the example-verified stages (test-mined / library-
    // behavior) that would solve correctly — losing the solve AND making the ladder flaky.
    // Require the synthesized program to reproduce the mined I/O; if it can't (and there ARE
    // asserts to check), decline so the proven stages run. No asserts -> keep prior behavior.
    if !synthesis_reproduces_failing_asserts(context, &repo_fn, &result.code) {
        return None;
    }
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
/// MODEL-FREE MUTATION REPAIR — the deterministic engine coding BEYOND pure-function synthesis. When
/// code already EXISTS but has a small bug the example-based synthesizer can't reach — a wrong
/// operator, an off-by-one, `=` where `+=` was meant, a bug in a struct METHOD (no `f(x)=y` pairs to
/// mine) — enumerate single-edit mutations of the existing NON-test code, apply each to the isolated
/// work copy, and run the repo's own test command. The first mutation that makes cargo test pass IS
/// the repair: no model, no I/O examples, just search + the compiler/test oracle. Bounded in count +
/// wall-clock so it stays a fast model-free tier; runs AFTER the synthesizers (which handle stubs)
/// and BEFORE the LLM lane.
pub fn try_mutation_repair_patch(
    task: &RepoTaskSpec,
    context: &RepairContext,
    analysis: Option<&FailureAnalysis>,
) -> Option<RepairPatch> {
    let verifier = RepairVerifier::new(&context.root, GuardrailPolicy::default());
    let preferred = analysis.and_then(|a| a.file.clone());
    let target_line = analysis.and_then(|a| a.line);
    let mut files: Vec<(String, String)> = context
        .files
        .iter()
        .filter(|f| f.path.ends_with(".rs"))
        .filter_map(|f| {
            let t = f.text.as_deref()?;
            t.contains("fn ").then(|| (f.path.clone(), t.to_string()))
        })
        .collect();
    // Mutate the failure-implicated file first.
    files.sort_by_key(|(p, _)| {
        !preferred.as_deref().map(|pf| pf.ends_with(p) || p.ends_with(pf)).unwrap_or(false)
    });
    let mk_patch = |path: &str, orig: &str, mutated: String| {
        Some(
            RepairPatch::new()
                .with_edit(RepairEdit::new(
                    path.to_string(),
                    orig.to_string(),
                    mutated,
                    "mutation repair (model-free; cargo-test gated)",
                ))
                .with_metadata("proposer", "mutation_repair"),
        )
    };
    let start = std::time::Instant::now();
    let budget = std::time::Duration::from_secs(90);
    const MAX_SINGLE: usize = 32;
    const MAX_BASES: usize = 8;
    const MAX_PAIR: usize = 24;
    let mut tried = 0usize;
    // The first file's mutations that COMPILE but still fail — bases for a second edit (the built-in
    // compile pre-filter: cargo test already told us which mutations type-check).
    let mut two_edit: Option<(String, String, std::path::PathBuf, Vec<String>)> = None;
    for (path, orig) in &files {
        let abs = std::path::Path::new(&context.root).join(path);
        let mut bases: Vec<String> = Vec::new();
        // LOCALIZE: if the failure names this file + a line (a panic `src/lib.rs:140`), mutate ONLY
        // the function containing that line first. A large file makes thousands of whole-file operator
        // mutations that bury the real fix past MAX_SINGLE; the failing function's ~dozens of mutations
        // fit under the cap and are precisely where the bug is.
        let span = if preferred
            .as_deref()
            .map(|pf| pf.ends_with(path.as_str()) || path.ends_with(pf))
            .unwrap_or(false)
        {
            target_line.and_then(|ln| production_target_span(orig, ln as usize))
        } else {
            None
        };
        // STUB GENERATION first (fill empty non-pure methods from struct-field templates), then
        // single-edit MUTATIONS of existing code (localized function first, then whole-file).
        let candidates = generate_stub_fills(orig)
            .into_iter()
            .chain(localized_then_full_mutations(orig, span));
        for mutated in candidates {
            if tried >= MAX_SINGLE || start.elapsed() > budget {
                break;
            }
            tried += 1;
            if std::fs::write(&abs, &mutated).is_err() {
                continue;
            }
            let v = verifier.verify(&task.test_command);
            let _ = std::fs::write(&abs, orig); // revert; the loop re-applies the winning patch
            match v {
                Ok(ver) if ver.success => return mk_patch(path, orig, mutated),
                // Compiled (no rustc error) but the test failed -> a candidate base for a 2nd edit.
                Ok(ver)
                    if bases.len() < MAX_BASES
                        && !ver.stderr.contains("error[")
                        && !ver.stderr.contains("could not compile") =>
                {
                    bases.push(mutated)
                }
                _ => {}
            }
        }
        if two_edit.is_none() && !bases.is_empty() {
            two_edit = Some((path.clone(), orig.clone(), abs.clone(), bases));
        }
        if tried >= MAX_SINGLE || start.elapsed() > budget {
            break;
        }
    }
    // TWO-EDIT search: for the non-pure multi-edit bug the single pass + the synthesizer both miss
    // (a struct method with two wrong tokens, no I/O pairs to mine). Layer a second single-edit
    // mutation onto each compiling base and cargo-test it. Tightly bounded; best-effort before the LLM.
    if let Some((path, orig, abs, bases)) = two_edit {
        let mut tried2 = 0usize;
        for base in &bases {
            for m2 in generate_mutations(base) {
                if tried2 >= MAX_PAIR || start.elapsed() > budget {
                    return None;
                }
                if m2 == orig {
                    continue; // a second edit that undoes the first
                }
                tried2 += 1;
                if std::fs::write(&abs, &m2).is_err() {
                    continue;
                }
                let passed = verifier.verify(&task.test_command).map(|v| v.success).unwrap_or(false);
                let _ = std::fs::write(&abs, &orig);
                if passed {
                    return mk_patch(&path, &orig, m2);
                }
            }
        }
    }
    None
}

/// Passing-test count summed across every `test result:` line in cargo's stdout — the score the
/// multi-hole search maximizes (a per-hole fill can turn its own test green while others stay red,
/// so all-or-nothing `success` is too coarse to drive coordinate descent).
fn passed_count(stdout: &str) -> usize {
    stdout
        .lines()
        .filter_map(|l| l.find(" passed").map(|i| &l[..i]))
        .filter_map(|prefix| prefix.split_whitespace().last())
        .filter_map(|n| n.parse::<usize>().ok())
        .sum()
}

/// PHASE 1 — MULTI-HOLE COORDINATION (model-free). A repo with SEVERAL empty stubs across a struct's
/// methods (and free fns) that only COMPILES once every hole is filled defeats single-hole proposers:
/// no one fill produces a compiling crate, so nothing verifies. This fills every hole with a
/// type-default (the compile floor -> tests run) then coordinate-descends: for each hole, try its
/// field-derived candidate bodies holding the others fixed, keep the fill that maximizes passing
/// tests, iterate until all green or no pass improves. Returns ONE multi-file patch. All cargo-gated,
/// no model. Fires only when >=2 holes exist (single-hole is handled by stub-gen in the mutation tier).
/// One `.rs` file with empty-body holes and the current body chosen for each.
struct FileHoles {
    path: String,
    abs: std::path::PathBuf,
    orig: String,
    code: String,
    tail: String,
    holes: Vec<Hole>,
    choice: Vec<String>,
}

impl FileHoles {
    /// The file with every hole's current `choice` spliced in (test module reattached).
    fn render(&self) -> String {
        let mut s = self.code.clone();
        for (h, body) in self.holes.iter().zip(&self.choice).rev() {
            s.replace_range(h.body_open..h.close, &format!(" {body} "));
        }
        s.push_str(&self.tail);
        s
    }
}

/// (all-pass, #passing-tests, compiled) from a cargo verification.
fn mh_score(v: &Result<RepairVerification, String>) -> (bool, usize, bool) {
    match v {
        Ok(r) => (
            r.success,
            passed_count(&r.stdout),
            !r.stderr.contains("error[") && !r.stderr.contains("could not compile"),
        ),
        Err(_) => (false, 0, false),
    }
}

/// Fingerprint of the current multi-hole fill choices (for verify-result caching).
fn mh_files_fingerprint(files: &[FileHoles]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for fh in files {
        fh.path.hash(&mut h);
        for c in &fh.choice {
            c.hash(&mut h);
        }
    }
    h.finish()
}

/// Cached `cargo test` score: joint search revisits identical fill states across
/// combo × descent passes; skipping a redundant verify is the main ~50s lever.
fn mh_score_cached(
    files: &[FileHoles],
    verifier: &RepairVerifier,
    cmd: &str,
    cache: &mut HashMap<u64, (bool, usize, bool)>,
) -> (bool, usize, bool) {
    let fp = mh_files_fingerprint(files);
    if let Some(&s) = cache.get(&fp) {
        return s;
    }
    let s = mh_score(&verifier.verify(cmd));
    cache.insert(fp, s);
    s
}

/// Bounded cartesian product of candidate lists (caps the number of combinations so the joint search
/// can't explode on a struct with many mutators).
fn bounded_product(lists: &[Vec<String>], cap: usize) -> Vec<Vec<String>> {
    let mut out: Vec<Vec<String>> = vec![vec![]];
    for list in lists {
        let mut next = Vec::new();
        'outer: for prefix in &out {
            for item in list {
                if next.len() >= cap {
                    break 'outer;
                }
                let mut v = prefix.clone();
                v.push(item.clone());
                next.push(v);
            }
        }
        out = next;
        if out.is_empty() {
            break;
        }
    }
    out
}

/// One coordinate-descent solve over the non-`fixed` holes: reset them to their defaults, then for
/// each try its candidates holding the rest fixed, keeping the fill that raises the passing-test
/// count (and committing a first-compiling guess to unstick a prerequisite whose effect is invisible
/// until a getter is filled). `fixed[fi][hi]` pins a hole (used by the joint search to hold a mutator
/// combo). Returns true on a full solve, leaving `files` AT the solution. Writes as it goes.
/// `cache` memoizes verify scores by fill fingerprint (joint-search perf).
#[allow(clippy::too_many_arguments)]
fn mh_descend(
    files: &mut [FileHoles],
    fixed: &[Vec<bool>],
    verifier: &RepairVerifier,
    cmd: &str,
    start: std::time::Instant,
    budget: std::time::Duration,
    runs: &mut usize,
    max_runs: usize,
    cache: &mut HashMap<u64, (bool, usize, bool)>,
) -> bool {
    for fi in 0..files.len() {
        for hi in 0..files[fi].holes.len() {
            if !fixed[fi][hi] {
                files[fi].choice[hi] = files[fi].holes[hi].default.clone();
            }
        }
    }
    for fh in files.iter() {
        let _ = std::fs::write(&fh.abs, fh.render());
    }
    let (ok, mut cur_passed, compiled) = mh_score_cached(files, verifier, cmd, cache);
    if ok {
        return true;
    }
    if !compiled {
        return false;
    }
    for _pass in 0..6 {
        let mut improved = false;
        for fi in 0..files.len() {
            for hi in 0..files[fi].holes.len() {
                if fixed[fi][hi] {
                    continue;
                }
                let cands = files[fi].holes[hi].candidates.clone();
                let default_body = files[fi].holes[hi].default.clone();
                let mut best_body = files[fi].choice[hi].clone();
                let mut best_passed = cur_passed;
                let mut first_guess: Option<String> = None;
                for cand in cands {
                    if *runs >= max_runs || start.elapsed() > budget {
                        files[fi].choice[hi] = best_body.clone();
                        let _ = std::fs::write(&files[fi].abs, files[fi].render());
                        return mh_score_cached(files, verifier, cmd, cache).0;
                    }
                    files[fi].choice[hi] = cand.clone();
                    if std::fs::write(&files[fi].abs, files[fi].render()).is_err() {
                        continue;
                    }
                    *runs += 1;
                    let (o, passed, comp) = mh_score_cached(files, verifier, cmd, cache);
                    if o {
                        return true;
                    }
                    if comp {
                        if first_guess.is_none() {
                            first_guess = Some(cand.clone());
                        }
                        if passed > best_passed {
                            best_passed = passed;
                            best_body = cand.clone();
                        }
                    }
                }
                if best_passed > cur_passed {
                    files[fi].choice[hi] = best_body;
                    cur_passed = best_passed;
                    improved = true;
                } else if best_body == default_body {
                    // No score lift: commit first compiling guess to unstick a
                    // prerequisite whose effect is invisible until a getter fills.
                    // (Was incorrectly checking choice==default after the cand loop
                    // left choice on the last candidate — first_guess never fired.)
                    if let Some(g) = first_guess {
                        files[fi].choice[hi] = g;
                        improved = true;
                    } else {
                        files[fi].choice[hi] = default_body;
                    }
                } else {
                    files[fi].choice[hi] = best_body;
                }
                let _ = std::fs::write(&files[fi].abs, files[fi].render());
            }
        }
        if !improved {
            break;
        }
    }
    mh_score_cached(files, verifier, cmd, cache).0
}

pub fn try_multihole_fill_patch(
    task: &RepoTaskSpec,
    context: &RepairContext,
    _analysis: Option<&FailureAnalysis>,
) -> Option<RepairPatch> {
    let mut files: Vec<FileHoles> = Vec::new();
    let mut total_holes = 0usize;
    for f in &context.files {
        if !f.path.ends_with(".rs") {
            continue;
        }
        let Some(text) = f.text.as_deref() else { continue };
        let ts = text.find("#[cfg(test)]").unwrap_or(text.len());
        let code = text[..ts].to_string();
        let holes = scan_holes(&code);
        if holes.is_empty() {
            continue;
        }
        total_holes += holes.len();
        let choice = holes.iter().map(|h| h.default.clone()).collect();
        files.push(FileHoles {
            path: f.path.clone(),
            abs: std::path::Path::new(&context.root).join(&f.path),
            orig: text.to_string(),
            tail: text[ts..].to_string(),
            code,
            holes,
            choice,
        });
    }
    if total_holes < 2 {
        return None; // single-hole: stub-gen in the mutation tier covers it
    }
    let verifier = RepairVerifier::new(&context.root, GuardrailPolicy::default());
    let revert_all = |files: &[FileHoles]| {
        for fh in files {
            let _ = std::fs::write(&fh.abs, &fh.orig);
        }
    };
    let finish = |files: &[FileHoles]| -> Option<RepairPatch> {
        let mut patch = RepairPatch::new().with_metadata("proposer", "multihole_fill");
        let mut changed = false;
        for fh in files {
            let rendered = fh.render();
            if rendered != fh.orig {
                patch = patch.with_edit(RepairEdit::new(
                    fh.path.clone(),
                    fh.orig.clone(),
                    rendered,
                    "multi-hole fill (model-free; cargo-test gated)",
                ));
                changed = true;
            }
        }
        changed.then_some(patch)
    };
    // Compile floor: write every hole at its default; the crate must now compile (tests run+fail).
    for fh in &files {
        if std::fs::write(&fh.abs, fh.render()).is_err() {
            revert_all(&files);
            return None;
        }
    }
    let (floor_ok, _, floor_compiled) = mh_score(&verifier.verify(&task.test_command));
    if floor_ok || !floor_compiled {
        // defaults already pass (let another tier own it) OR type-defaults don't compile (unsupported)
        revert_all(&files);
        return None;
    }
    let start = std::time::Instant::now();
    let budget = std::time::Duration::from_secs(150);
    let mut runs = 0usize;
    const MAX_RUNS: usize = 400;
    let cmd = &task.test_command;
    let mut cache: HashMap<u64, (bool, usize, bool)> = HashMap::new();

    // Phase 1: plain coordinate descent (no pinned holes).
    let none_fixed: Vec<Vec<bool>> = files.iter().map(|f| vec![false; f.holes.len()]).collect();
    if mh_descend(
        &mut files,
        &none_fixed,
        &verifier,
        cmd,
        start,
        budget,
        &mut runs,
        MAX_RUNS,
        &mut cache,
    ) {
        let patch = finish(&files);
        revert_all(&files);
        return patch;
    }

    // Phase 1.5: JOINT search over prerequisite (mutator) combos. On a multi-field struct the plain
    // descent gets stuck in a local minimum — a prerequisite mutator's first-candidate guess targets
    // the wrong field and no single move escapes. Pin each mutator to a candidate combination and let
    // the getters descend against it; the right combo (e.g. deposit->push, charge_fee->fees=f) makes
    // the getter tests reachable. Bounded product so it can't explode.
    let prereqs: Vec<(usize, usize)> = files
        .iter()
        .enumerate()
        .flat_map(|(fi, f)| {
            f.holes
                .iter()
                .enumerate()
                .filter(|(_, h)| h.is_prereq && h.candidates.len() > 1)
                .map(move |(hi, _)| (fi, hi))
        })
        .collect();
    if !prereqs.is_empty() {
        let lists: Vec<Vec<String>> = prereqs
            .iter()
            .map(|(fi, hi)| {
                let mut c = files[*fi].holes[*hi].candidates.clone();
                c.truncate(6); // keep the guarded-mutation candidates, not just the 4 plain op-assigns
                c
            })
            .collect();
        let fixed: Vec<Vec<bool>> = files
            .iter()
            .enumerate()
            .map(|(fi, f)| (0..f.holes.len()).map(|hi| prereqs.contains(&(fi, hi))).collect())
            .collect();
        for combo in bounded_product(&lists, 40) {
            if runs >= MAX_RUNS || start.elapsed() > budget {
                break;
            }
            for ((fi, hi), body) in prereqs.iter().zip(&combo) {
                files[*fi].choice[*hi] = body.clone();
            }
            if mh_descend(
                &mut files,
                &fixed,
                &verifier,
                cmd,
                start,
                budget,
                &mut runs,
                MAX_RUNS,
                &mut cache,
            ) {
                let patch = finish(&files);
                revert_all(&files);
                return patch;
            }
        }
    }
    revert_all(&files);
    None
}

/// MODEL-FREE STUB GENERATION for non-pure methods (coding ARBITRARY programs, not just repairing):
/// a `&mut self` method or getter with an EMPTY body — which the example-based synthesizer can't
/// reach (no `f(x)=y` pairs) and mutation can't touch (nothing to edit) — is filled from templates
/// built out of the struct's own fields: `self.F = P;`, `self.F += P;`, `self.F.push(P)`, `self.F +=
/// 1;`, `return self.F;`, etc. Each candidate is cargo-tested; the one that passes IS the
/// implementation. Bounded; enumerated over (field x param x template).
/// An empty-body method/function hole: its body span in `code` (`[body_open, close)`, between the
/// braces), a type-correct DEFAULT body that compiles (for the multi-hole compile floor), and the
/// field-derived candidate bodies to try.
struct Hole {
    body_open: usize,
    close: usize,
    default: String,
    candidates: Vec<String>,
    /// A pure state mutator (`&mut self`, no return) — a PREREQUISITE whose effect is only visible
    /// through a getter, so it must be pinned during the joint search that escapes coordinate-descent
    /// local minima on multi-field structs.
    is_prereq: bool,
}

/// Every `struct NAME { .. }` in `code` as (name, [(field, type)]).
fn parse_structs(code: &str) -> Vec<(String, Vec<(String, String)>)> {
    let mut out = Vec::new();
    let cb = code.as_bytes();
    let mut search = 0;
    while let Some(rel) = code[search..].find("struct ") {
        let at = search + rel;
        search = at + 7;
        if at > 0 && (cb[at - 1].is_ascii_alphanumeric() || cb[at - 1] == b'_') {
            continue; // not a word-boundary `struct`
        }
        let after = &code[at + 7..];
        let name_end = after.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(after.len());
        let name = after[..name_end].to_string();
        let Some(open) = after.find('{') else { continue };
        let ab = after.as_bytes();
        let mut depth = 0i32;
        let mut close = None;
        for i in open..ab.len() {
            match ab[i] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(close) = close else { continue };
        if name.is_empty() {
            continue;
        }
        let fields = parse_typed_params(&after[open + 1..close])
            .into_iter()
            .map(|(n, t)| (n.trim_start_matches("pub ").trim().to_string(), t))
            .filter(|(n, _)| !n.is_empty())
            .collect();
        out.push((name, fields));
    }
    out
}

/// Each `impl [Trait for] NAME { .. }` block as (self-type-name, body_open_byte, body_close_byte),
/// so a method's `self` fields resolve to the RIGHT struct in a multi-struct file.
fn impl_blocks(code: &str) -> Vec<(String, usize, usize)> {
    let mut out = Vec::new();
    let cb = code.as_bytes();
    let mut search = 0;
    while let Some(rel) = code[search..].find("impl").map(|r| search + r) {
        let at = rel;
        search = at + 4;
        let before_ok = at == 0 || !(cb[at - 1].is_ascii_alphanumeric() || cb[at - 1] == b'_');
        let after_ok = matches!(cb.get(at + 4), Some(b' ') | Some(b'<'));
        if !before_ok || !after_ok {
            continue;
        }
        let after = &code[at..];
        let Some(open) = after.find('{') else { break };
        let header = &after[4..open];
        let ty = header.rsplit(" for ").next().unwrap_or(header).trim();
        let name: String = ty.trim_start_matches('&').chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
        if name.is_empty() {
            continue;
        }
        let ab = after.as_bytes();
        let mut depth = 0i32;
        let mut close = None;
        for i in open..ab.len() {
            match ab[i] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(close) = close else { continue };
        out.push((name, at + open, at + close));
    }
    out
}

/// Scan `code` (the non-test part) for every function/method with an EMPTY body and build, per hole,
/// its candidate bodies plus a type-default. STRUCT-AWARE: a method's candidates use ITS impl's
/// struct fields (right struct in a multi-struct file), and a `Vec<Record>` collection field unlocks
/// typed-record templates — construct-and-push (`self.items.push(Item { title, priority })`) and
/// field-aggregate getters (`self.items.iter().map(|e| e.priority).sum()`). Shared by
/// generate_stub_fills (single-hole) and try_multihole_fill_patch (coordinate-descent over all holes).
fn scan_holes(code: &str) -> Vec<Hole> {
    let structs = parse_structs(code);
    let impls = impl_blocks(code);
    let fields_of = |name: &str| -> Vec<(String, String)> {
        structs.iter().find(|(n, _)| n == name).map(|(_, f)| f.clone()).unwrap_or_default()
    };
    let elem_type = |vec_ty: &str| -> Option<String> {
        vec_ty.strip_prefix("Vec<").and_then(|s| s.strip_suffix('>')).map(|s| s.trim().to_string())
    };
    let mut holes = Vec::new();
    let cb = code.as_bytes();
    let mut search = 0;
    while let Some(rel) = code[search..].find("fn ") {
        let fstart = search + rel;
        search = fstart + 3;
        let after = &code[fstart..];
        let Some(op) = after.find('(') else { continue };
        let ab = after.as_bytes();
        let mut depth = 0i32;
        let mut cp = None;
        for i in op..ab.len() {
            match ab[i] {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        cp = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(cp) = cp else { continue };
        let params_str = &after[op + 1..cp];
        let post = &after[cp + 1..];
        let (ret, brace_rel) = match (post.find("->"), post.find('{')) {
            (Some(a), Some(bpos)) if a < bpos => (post[a + 2..bpos].trim().to_string(), bpos),
            (_, Some(bpos)) => (String::new(), bpos),
            _ => continue,
        };
        let body_open = fstart + cp + 1 + brace_rel + 1; // just after `{`
        let mut d = 1i32;
        let mut close = None;
        let mut k = body_open;
        while k < cb.len() {
            match cb[k] {
                b'{' => d += 1,
                b'}' => {
                    d -= 1;
                    if d == 0 {
                        close = Some(k);
                        break;
                    }
                }
                _ => {}
            }
            k += 1;
        }
        let Some(close) = close else { continue };
        if !code[body_open..close].trim().is_empty() {
            continue; // not an empty stub
        }
        // Resolve `self` to the impl's struct (right fields in a multi-struct file).
        let self_fields: Vec<(String, String)> = impls
            .iter()
            .find(|(_, o, c)| body_open > *o && body_open < *c)
            .map(|(n, _, _)| fields_of(n))
            .unwrap_or_default();
        let fields = &self_fields;
        let is_mut = params_str.contains("&mut self");
        let value_typed: Vec<(String, String)> = parse_typed_params(params_str)
            .into_iter()
            .filter(|(n, _)| n != "self")
            .collect();
        let value_params: Vec<String> = value_typed.iter().map(|(n, _)| n.clone()).collect();
        let has_ret = !ret.is_empty() && ret != "()";
        let has_self = params_str.contains("self");
        let scalar_fields: Vec<&String> = fields.iter().filter(|(_, t)| t == "i64").map(|(n, _)| n).collect();
        let mut bodies: Vec<String> = Vec::new();
        // FREE FUNCTION hole (no self): a pure combination of its VALUE params — `net(balance,fees) ->
        // balance - fees`. Light arithmetic templates so a pure fn coupled with struct-method holes
        // (same crate won't compile until all are filled) is fillable in the same coordinate descent;
        // standalone pure fns are still handled by the full synthesizer tier before this one.
        if !has_self && has_ret && !value_params.is_empty() {
            let ps = &value_params;
            if ret == "i64" {
                for p in ps {
                    bodies.push(format!("return {p};"));
                }
                for i in 0..ps.len() {
                    for j in 0..ps.len() {
                        if i == j {
                            continue;
                        }
                        for op in ["+", "-", "*"] {
                            bodies.push(format!("return {} {op} {};", ps[i], ps[j]));
                        }
                    }
                }
            } else if ret == "bool" && ps.len() >= 2 {
                for op in ["==", "!=", "<", ">", "<=", ">="] {
                    bodies.push(format!("return {} {op} {};", ps[0], ps[1]));
                }
            }
        }
        if has_self && has_ret {
            for (fname, fty) in fields {
                if type_compatible(fty, &ret) {
                    bodies.push(format!("return self.{fname};"));
                }
                if ret == "i64" && (fty.starts_with("Vec<") || fty == "String") {
                    bodies.push(format!("return self.{fname}.len() as i64;"));
                }
                // Vec<i64> field -> a scalar AGGREGATE (sum/max/min/first/last) — the common
                // "total()/most_expensive()/cheapest()" getters over a collection field.
                if ret == "i64" && fty == "Vec<i64>" {
                    bodies.push(format!("self.{fname}.iter().sum()"));
                    bodies.push(format!("self.{fname}.iter().copied().max().unwrap_or(0)"));
                    bodies.push(format!("self.{fname}.iter().copied().min().unwrap_or(0)"));
                    bodies.push(format!("*self.{fname}.first().unwrap_or(&0)"));
                    bodies.push(format!("*self.{fname}.last().unwrap_or(&0)"));
                }
            }
            if ret == "i64" {
                for a in 0..scalar_fields.len() {
                    for b in 0..scalar_fields.len() {
                        if a == b {
                            continue;
                        }
                        for op in ["+", "-", "*"] {
                            bodies.push(format!("return self.{} {op} self.{};", scalar_fields[a], scalar_fields[b]));
                        }
                    }
                }
                // Vec<Record> field -> aggregate over one of the record's i64 fields
                // (`self.items.iter().map(|e| e.priority).sum()/max/min`).
                for (fname, fty) in fields {
                    if let Some(elem) = elem_type(fty) {
                        for (rf, rft) in fields_of(&elem) {
                            if rft == "i64" {
                                bodies.push(format!("self.{fname}.iter().map(|e| e.{rf}).sum()"));
                                bodies.push(format!("self.{fname}.iter().map(|e| e.{rf}).max().unwrap_or(0)"));
                                bodies.push(format!("self.{fname}.iter().map(|e| e.{rf}).min().unwrap_or(0)"));
                            }
                        }
                    }
                }
            }
            // CRUD READS. A bool getter over a collection field: is_empty / non-empty.
            if ret == "bool" {
                for (fname, fty) in fields {
                    if fty.starts_with("Vec<") || fty == "String" {
                        bodies.push(format!("return self.{fname}.is_empty();"));
                        bodies.push(format!("return !self.{fname}.is_empty();"));
                    }
                }
            }
            // LOOKUP BY KEY: a single String param selects a record by a String field, then either
            // reports existence (`contains(name) -> bool`) or returns one of its i64 fields
            // (`price_of(name) -> i64`). The everyday "query the collection by its key" operation.
            if value_typed.len() == 1 && value_typed[0].1 == "String" {
                let key = &value_typed[0].0;
                for (fname, fty) in fields {
                    if let Some(elem) = elem_type(fty) {
                        let ef = fields_of(&elem);
                        for (sf, sft) in &ef {
                            if sft != "String" {
                                continue;
                            }
                            if ret == "bool" {
                                bodies.push(format!("return self.{fname}.iter().any(|e| e.{sf} == {key});"));
                            }
                            if ret == "i64" {
                                for (rf, rft) in &ef {
                                    if rft == "i64" {
                                        bodies.push(format!("self.{fname}.iter().find(|e| e.{sf} == {key}).map(|e| e.{rf}).unwrap_or(0)"));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // INDEXED read: a single i64 param addresses an element (`price_at(i) -> items[i].price`,
            // `get(i) -> items[i].clone()`, or a scalar `Vec<i64>` element).
            if value_typed.len() == 1 && value_typed[0].1 == "i64" {
                let idx = &value_typed[0].0;
                for (fname, fty) in fields {
                    if fty == "Vec<i64>" && ret == "i64" {
                        bodies.push(format!("return self.{fname}[{idx} as usize];"));
                    }
                    if let Some(elem) = elem_type(fty) {
                        if ret == elem {
                            bodies.push(format!("return self.{fname}[{idx} as usize].clone();"));
                        }
                        if ret == "i64" {
                            for (rf, rft) in fields_of(&elem) {
                                if rft == "i64" {
                                    bodies.push(format!("return self.{fname}[{idx} as usize].{rf};"));
                                }
                            }
                        }
                    }
                }
            }
        }
        if is_mut {
            // AUTH-GATED mutation (access control as verified behavior): a param whose NAME matches a
            // struct field is a CREDENTIAL; gate the mutation on it so the wrong credential is a no-op
            // (`withdraw(amount, pin) { if pin == self.pin && amount <= self.balance { self.balance -=
            // amount; } }`). Emitted FIRST so the credential form survives the joint-search truncation.
            let field_names: Vec<&str> = fields.iter().map(|(n, _)| n.as_str()).collect();
            let creds: Vec<&str> =
                value_typed.iter().filter(|(n, t)| t == "i64" && field_names.contains(&n.as_str())).map(|(n, _)| n.as_str()).collect();
            let amounts: Vec<&str> =
                value_typed.iter().filter(|(n, t)| t == "i64" && !field_names.contains(&n.as_str())).map(|(n, _)| n.as_str()).collect();
            for c in &creds {
                for (f, fty) in fields {
                    if fty != "i64" || f == c {
                        continue; // target a DIFFERENT scalar field than the credential
                    }
                    for a in &amounts {
                        bodies.push(format!("if {c} == self.{c} && {a} <= self.{f} {{ self.{f} -= {a}; }}"));
                        bodies.push(format!("if {c} == self.{c} {{ self.{f} -= {a}; }}"));
                        bodies.push(format!("if {c} == self.{c} {{ self.{f} += {a}; }}"));
                    }
                }
            }
            for (fname, fty) in fields {
                if let Some(elem) = elem_type(fty) {
                    let rf = fields_of(&elem);
                    if rf.is_empty() {
                        // primitive element (Vec<i64>/Vec<String>): push each param directly.
                        for p in &value_params {
                            bodies.push(format!("self.{fname}.push({p});"));
                        }
                    } else if !value_params.is_empty() && rf.len() == value_params.len() {
                        // record element: CONSTRUCT the record from the params, then push.
                        let inits = rf
                            .iter()
                            .zip(&value_params)
                            .map(|((rn, _), p)| format!("{rn}: {p}"))
                            .collect::<Vec<_>>()
                            .join(", ");
                        bodies.push(format!("self.{fname}.push({elem} {{ {inits} }});"));
                    }
                } else {
                    for p in &value_params {
                        for op in ["=", "+=", "-=", "*="] {
                            bodies.push(format!("self.{fname} {op} {p};"));
                        }
                    }
                    // GUARDED mutation: conditional op-assign for the common "rules" -- no-overdraft
                    // (`if amount <= self.balance {{ self.balance -= amount; }}`), keep-max / keep-min
                    // (`if p > self.f {{ self.f = p; }}`). This is the logic a schema can't decide, so
                    // it's the reach that makes Phase 3 fills real.
                    if fty == "i64" {
                        for (p, pt) in &value_typed {
                            if pt == "i64" {
                                bodies.push(format!("if {p} <= self.{fname} {{ self.{fname} -= {p}; }}"));
                                bodies.push(format!("if {p} > self.{fname} {{ self.{fname} = {p}; }}"));
                                bodies.push(format!("if {p} < self.{fname} {{ self.{fname} = {p}; }}"));
                            }
                        }
                    }
                    if value_params.is_empty() {
                        bodies.push(format!("self.{fname} += 1;"));
                        bodies.push(format!("self.{fname} -= 1;"));
                    }
                }
            }
            if value_params.len() >= 2 && scalar_fields.len() >= value_params.len() {
                for op in ["=", "+=", "-="] {
                    let stmts: String = value_params
                        .iter()
                        .enumerate()
                        .map(|(k, p)| format!("self.{} {op} {p}; ", scalar_fields[k]))
                        .collect();
                    bodies.push(stmts.trim_end().to_string());
                }
            }
            if has_ret && ret == "i64" {
                for f in &scalar_fields {
                    bodies.push(format!("self.{f} += 1; return self.{f};"));
                    for p in &value_params {
                        bodies.push(format!("self.{f} += {p}; return self.{f};"));
                    }
                }
            }
            // CRUD WRITES over a collection field: clear (no params), remove-by-index (one i64),
            // and field-update-by-index (`set_price(i, p) -> items[i].price = p`).
            for (fname, fty) in fields {
                if !fty.starts_with("Vec<") {
                    continue;
                }
                if value_typed.is_empty() {
                    bodies.push(format!("self.{fname}.clear();"));
                }
                if value_typed.len() == 1 && value_typed[0].1 == "i64" {
                    bodies.push(format!("self.{fname}.remove({} as usize);", value_typed[0].0));
                }
                if value_typed.len() == 2 && value_typed[0].1 == "i64" {
                    if let Some(elem) = elem_type(fty) {
                        let (idx, _) = &value_typed[0];
                        let (val, vty) = &value_typed[1];
                        for (rf, rft) in fields_of(&elem) {
                            if &rft == vty {
                                bodies.push(format!("self.{fname}[{idx} as usize].{rf} = {val};"));
                            }
                            // Record-field OPERATIONS WITH RULES: change a record's i64 field by index,
                            // guarded (`sell(i, qty) { if qty <= items[i].stock { items[i].stock -= qty; } }`)
                            // or plain (`restock(i, qty) { items[i].stock += qty; }`). Guarded first so it
                            // survives the joint-search truncation.
                            if rft == "i64" && vty == "i64" {
                                let slot = format!("self.{fname}[{idx} as usize].{rf}");
                                bodies.push(format!("if {val} <= {slot} {{ {slot} -= {val}; }}"));
                                bodies.push(format!("{slot} -= {val};"));
                                bodies.push(format!("{slot} += {val};"));
                            }
                        }
                    }
                }
            }
        }
        holes.push(Hole {
            body_open,
            close,
            default: type_default(&ret),
            candidates: bodies,
            is_prereq: is_mut && !has_ret,
        });
    }
    holes
}

/// A type-correct placeholder body (no trailing statement terminator) so an empty hole COMPILES —
/// the compile floor the multi-hole search stands on. `()` returns get an empty body.
fn type_default(ret: &str) -> String {
    match ret {
        "" | "()" => String::new(),
        "i64" | "i32" | "u64" | "usize" | "u32" => "0".to_string(),
        "bool" => "false".to_string(),
        "String" => "String::new()".to_string(),
        "f64" | "f32" => "0.0".to_string(),
        t if t.starts_with("Vec<") => "Vec::new()".to_string(),
        _ => "Default::default()".to_string(),
    }
}

fn generate_stub_fills(content: &str) -> Vec<String> {
    let test_start = content.find("#[cfg(test)]").unwrap_or(content.len());
    let code = &content[..test_start];
    let tail = &content[test_start..];
    if parse_structs(code).is_empty() {
        return Vec::new(); // single-hole stub-gen is for struct methods; free fns go to the synthesizer
    }
    let mut out = Vec::new();
    for h in scan_holes(code) {
        for body in &h.candidates {
            let mut m = String::with_capacity(content.len() + body.len() + 4);
            m.push_str(&code[..h.body_open]);
            m.push(' ');
            m.push_str(body);
            m.push(' ');
            m.push_str(&code[h.close..]);
            m.push_str(tail);
            out.push(m);
        }
    }
    out
}

/// `pub? name: TYPE` fields of every `struct NAME { .. }` in `code`.
fn parse_struct_fields(code: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut search = 0;
    while let Some(rel) = code[search..].find("struct ") {
        let at = search + rel;
        search = at + 7;
        let after = &code[at..];
        let Some(open) = after.find('{') else { continue };
        let ab = after.as_bytes();
        let mut depth = 0i32;
        let mut close = None;
        for i in open..ab.len() {
            match ab[i] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(close) = close else { continue };
        for (nm, ty) in parse_typed_params(&after[open + 1..close]) {
            let nm = nm.trim_start_matches("pub ").trim().to_string();
            if !nm.is_empty() {
                out.push((nm, ty));
            }
        }
    }
    out
}

/// Loose field-type vs return-type compatibility for a getter body.
fn type_compatible(field_ty: &str, ret: &str) -> bool {
    field_ty == ret
        || (field_ty == "i64" && ret == "i64")
        || (field_ty == "bool" && ret == "bool")
        || (field_ty == "String" && ret == "String")
}

/// Byte span `[start, end)` of the `fn` whose brace-balanced body contains the 1-indexed `line`.
/// Used to localize mutation search to the failing function (a panic reports `file:line`). Returns
/// the innermost enclosing function (last match wins). `None` if the line is out of range or no `fn`
/// encloses it (e.g. the line is a top-level `const`).
fn function_span_at_line(code: &str, line: usize) -> Option<(usize, usize)> {
    if line == 0 {
        return None;
    }
    // Byte offset of the start of the target line.
    let mut off = None;
    let mut ln = 1usize;
    if line == 1 {
        off = Some(0);
    }
    for (i, c) in code.char_indices() {
        if c == '\n' {
            ln += 1;
            if ln == line {
                off = Some(i + 1);
                break;
            }
        }
    }
    let off = off?;
    let cb = code.as_bytes();
    let mut search = 0usize;
    let mut best: Option<(usize, usize)> = None;
    while let Some(rel) = code[search..].find("fn ") {
        let fstart = search + rel;
        search = fstart + 3;
        // Only treat `fn ` as a definition when it is at a token boundary.
        if fstart > 0 {
            let prev = cb[fstart - 1];
            if prev.is_ascii_alphanumeric() || prev == b'_' {
                continue;
            }
        }
        let Some(brel) = code[fstart..].find('{') else {
            continue;
        };
        let bopen = fstart + brel;
        let mut depth = 0i32;
        let mut close = None;
        let mut k = bopen;
        while k < cb.len() {
            match cb[k] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(k);
                        break;
                    }
                }
                _ => {}
            }
            k += 1;
        }
        let Some(close) = close else { continue };
        if fstart <= off && off <= close {
            best = Some((fstart, close + 1));
        }
    }
    best
}

/// The production function to localize the mutation search to, given the failing `line`. If the line
/// is inside a production function (a panic-in-code bug), that function. If it is inside the
/// `#[cfg(test)]` module (an `assert_eq!` failure reports the ASSERT's line, not the code's), the
/// production function NAMED in that assertion — so an assert bug still localizes to the code UNDER
/// TEST, never the test, and never a sibling like a reference impl the assert also calls.
fn production_target_span(code: &str, line: usize) -> Option<(usize, usize)> {
    let test_start = code.find("#[cfg(test)]").unwrap_or(code.len());
    if let Some((s, e)) = function_span_at_line(code, line) {
        if s < test_start {
            return Some((s, e));
        }
    }
    named_production_fn_span(code, line, test_start)
}

/// Span of the first production function whose name is CALLED on the given (1-indexed) line.
fn named_production_fn_span(code: &str, line: usize, test_start: usize) -> Option<(usize, usize)> {
    let text = code.lines().nth(line.checked_sub(1)?)?;
    for name in called_idents(text) {
        if let Some(span) = fn_def_span_by_name(&code[..test_start], &name) {
            return Some(span);
        }
    }
    None
}

/// Free-function call names on a line: `NAME(` where `NAME` is neither a macro (`NAME!(` — the `!`
/// breaks the identifier walk) nor a method call (`.NAME(`). In source order, deduped.
fn called_idents(line: &str) -> Vec<String> {
    let b = line.as_bytes();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'(' && i > 0 {
            let mut j = i;
            while j > 0 && (b[j - 1].is_ascii_alphanumeric() || b[j - 1] == b'_') {
                j -= 1;
            }
            if j < i {
                let name = &line[j..i];
                let prev = if j > 0 { b[j - 1] } else { 0 };
                let starts_ident = name.as_bytes()[0].is_ascii_alphabetic() || name.as_bytes()[0] == b'_';
                if prev != b'.' && starts_ident && !out.iter().any(|s| s == name) {
                    out.push(name.to_string());
                }
            }
        }
        i += 1;
    }
    out
}

/// Byte span `[start, end)` of a top-level `fn NAME(...) { ... }` definition by name, brace-matched.
fn fn_def_span_by_name(code: &str, name: &str) -> Option<(usize, usize)> {
    let needle = format!("fn {name}");
    let cb = code.as_bytes();
    let mut search = 0usize;
    while let Some(rel) = code[search..].find(&needle) {
        let at = search + rel;
        search = at + needle.len();
        // token boundary before `fn`
        if at > 0 {
            let p = cb[at - 1];
            if p.is_ascii_alphanumeric() || p == b'_' {
                continue;
            }
        }
        // token boundary after the name: `(`, `<` (generics), or whitespace
        match code[at + needle.len()..].chars().next() {
            Some('(') | Some('<') | Some(' ') | Some('\t') => {}
            _ => continue,
        }
        let Some(brel) = code[at..].find('{') else { continue };
        let bopen = at + brel;
        let mut depth = 0i32;
        let mut k = bopen;
        while k < cb.len() {
            match cb[k] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some((at, k + 1));
                    }
                }
                _ => {}
            }
            k += 1;
        }
    }
    None
}

/// Mutations to try in order: first the localized function's variants (spliced back into the full
/// file), then the whole-file variants as a fallback. When `span` is `None` this is just the
/// whole-file mutations, so the caller path is unchanged when no line is known.
fn localized_then_full_mutations(orig: &str, span: Option<(usize, usize)>) -> Vec<String> {
    let mut out = Vec::new();
    // SOUNDNESS: never localize INTO the `#[cfg(test)]` module. An `assert_eq!` failure panics at the
    // ASSERT line, so the failing `file:line` points into the test — localizing there would mutate the
    // test's asserts and "fix" the failure by CORRUPTING the test (a never-wrong violation). The
    // whole-file pass below already excludes the test module; when the located span lies in the test
    // region, drop it and fall back to whole-file (which for an assert bug still reaches the buggy
    // production code). Localization only helps PANIC-in-production bugs, whose span is before this.
    let test_start = orig.find("#[cfg(test)]").unwrap_or(orig.len());
    if let Some((s, e)) = span {
        if s < test_start {
            let func = &orig[s..e];
            for fv in generate_mutations(func) {
                // `func` is production code with no `#[cfg(test)]`, so each `fv` is the mutated
                // function verbatim; splice it back into the surrounding file.
                out.push(format!("{}{}{}", &orig[..s], fv, &orig[e..]));
            }
        }
    }
    out.extend(generate_mutations(orig));
    out
}

/// Splice `to` in place of `from_len` bytes at `at` in `code`, then reappend `tail` (the test module).
fn splice_mutation(code: &str, tail: &str, at: usize, from_len: usize, to: &str) -> String {
    let mut m = String::with_capacity(code.len() + tail.len() + 2);
    m.push_str(&code[..at]);
    m.push_str(to);
    m.push_str(&code[at + from_len..]);
    m.push_str(tail);
    m
}

/// Single-edit mutations of the NON-test code: operator swaps, `=`->`+=`/`-=`, integer +-1. One
/// changed occurrence per candidate. The `#[cfg(test)]` module is left untouched (never "fix" the
/// test to make it pass).
fn generate_mutations(content: &str) -> Vec<String> {
    let test_start = content.find("#[cfg(test)]").unwrap_or(content.len());
    let code = &content[..test_start];
    let tail = &content[test_start..];
    let mut out = Vec::new();
    // UNDERFLOW GUARD (real-bug class, model-FREE) — emitted FIRST (high value, and a big file yields
    // thousands of operator mutations that would otherwise bury it past the search cap). A usize local
    // used later as `VAR - 1` panics when it is 0 (an index/exponent computed from a float, as in
    // bytesize's unit selection). For each `let VAR = <expr>;` whose VAR appears in `VAR - 1` afterwards,
    // clamp it: `let mut VAR = <expr>; if VAR == 0 { VAR = 1; }` — the only correct fix (a `saturating_sub`
    // on the index alone leaves an accompanying `pow(unit, VAR)` division wrong). Bounded.
    {
        let ub = code.as_bytes();
        let mut scan = 0usize;
        let mut made = 0usize;
        while let Some(rel) = code[scan..].find("let ") {
            let at = scan + rel;
            scan = at + 4;
            if at > 0 && (ub[at - 1].is_ascii_alphanumeric() || ub[at - 1] == b'_') {
                continue;
            }
            let after = &code[at + 4..];
            if after.trim_start().starts_with("mut ") {
                continue;
            }
            let name: String = after
                .trim_start()
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if name.is_empty() {
                continue;
            }
            let Some(eq_rel) = after.find('=') else { continue };
            let Some(semi_rel) = after[eq_rel..].find(';') else { continue };
            let semi = at + 4 + eq_rel + semi_rel;
            let rest = &code[semi..];
            if !(rest.contains(&format!("{name} - 1")) || rest.contains(&format!("{name}-1"))) {
                continue;
            }
            let stmt_tail = &code[at + 4..=semi];
            let mutated = format!("let mut {stmt_tail} if {name} == 0 {{ {name} = 1; }}");
            out.push(splice_mutation(code, tail, at, semi + 1 - at, &mutated));
            made += 1;
            if made >= 6 {
                break;
            }
        }
    }
    // Space-INSENSITIVE binary-operator swaps. Real Rust has `a+b`, `i<n`, `self.n=x` as often as the
    // spaced forms; a spaced-string match (` = `) silently MISSES the unspaced ones (this is why an
    // unspaced `self.n=x` off-by-op bug went unrepaired). Scan char by char, classify each operator
    // token with boundary checks (compound ops `==`/`+=`/`->`/`=>`, generics `Vec<T>`, unary `-1`,
    // deref `*x`, trait `->` all excluded), and replace only the operator so spacing is preserved.
    let cb = code.as_bytes();
    let eff_prev = |i: usize| -> u8 {
        let mut j = i;
        while j > 0 {
            j -= 1;
            if cb[j] != b' ' && cb[j] != b'\t' {
                return cb[j];
            }
        }
        0
    };
    let operand_end = |b: u8| b.is_ascii_alphanumeric() || b == b')' || b == b']' || b == b'_';
    let mut i = 0usize;
    while i < code.len() {
        let c = cb[i];
        let n1 = cb.get(i + 1).copied().unwrap_or(0);
        let p1 = if i > 0 { cb[i - 1] } else { 0 };
        let sp = p1 == b' ' || n1 == b' '; // a real comparison is usually spaced; a generic `<` never is
        let (len, reps): (usize, &[&str]) = if c == b'<' && n1 == b'=' {
            (2, &["<", ">="])
        } else if c == b'>' && n1 == b'=' {
            (2, &[">", "<="])
        } else if c == b'=' && n1 == b'=' {
            (2, &["!="])
        } else if c == b'!' && n1 == b'=' {
            (2, &["=="])
        } else if c == b'+' && n1 == b'=' && p1 != b'+' {
            (2, &["-=", "*="]) // compound-assign swap: `total += x` <-> `-=` (wrong-op struct bug)
        } else if c == b'-' && n1 == b'=' && p1 != b'-' && p1 != b'>' {
            (2, &["+="])
        } else if c == b'*' && n1 == b'=' && p1 != b'*' {
            (2, &["+=", "/="])
        } else if c == b'<' && n1 != b'=' && n1 != b'<' && p1 != b'<' && sp {
            (1, &["<=", ">"])
        } else if c == b'>' && n1 != b'=' && n1 != b'>' && p1 != b'>' && p1 != b'-' && p1 != b'=' && sp {
            (1, &[">=", "<"])
        } else if c == b'='
            && n1 != b'='
            && n1 != b'>'
            && !matches!(p1, b'=' | b'<' | b'>' | b'!' | b'+' | b'-' | b'*' | b'/' | b'%')
        {
            (1, &["+=", "-="])
        } else if c == b'+' && n1 != b'=' && n1 != b'+' && p1 != b'+' && operand_end(eff_prev(i)) {
            (1, &["-", "*"])
        } else if c == b'-' && n1 != b'=' && n1 != b'>' && n1 != b'-' && p1 != b'-' && operand_end(eff_prev(i)) {
            (1, &["+"])
        } else if c == b'*' && n1 != b'=' && n1 != b'*' && p1 != b'*' && operand_end(eff_prev(i)) {
            (1, &["+", "/"])
        } else if c == b'/' && n1 != b'=' && n1 != b'/' && n1 != b'*' && p1 != b'/' && operand_end(eff_prev(i)) {
            (1, &["*"])
        } else if c == b'%' && n1 != b'=' && operand_end(eff_prev(i)) {
            (1, &["/"])
        } else {
            (0, &[])
        };
        if len > 0 {
            for to in reps {
                out.push(splice_mutation(code, tail, i, len, to));
            }
            i += len;
        } else {
            i += 1;
        }
    }
    // METHOD-NAME swaps: a wrong stdlib method call (`.to_lowercase()` should be `.to_uppercase()`,
    // `.first()` vs `.last()`, `.min()` vs `.max()`, `.pop()` vs `.remove(0)`, iterate forward vs
    // reversed). Whole-token replacement wherever it appears.
    const METHOD_SWAPS: &[(&str, &str)] = &[
        (".to_uppercase()", ".to_lowercase()"),
        (".to_lowercase()", ".to_uppercase()"),
        (".first()", ".last()"),
        (".last()", ".first()"),
        (".min()", ".max()"),
        (".max()", ".min()"),
        (".iter()", ".iter().rev()"),
        (".pop()", ".remove(0)"),
    ];
    for (from, to) in METHOD_SWAPS {
        let mut idx = 0;
        while let Some(pos) = code[idx..].find(from) {
            let at = idx + pos;
            out.push(splice_mutation(code, tail, at, from.len(), to));
            idx = at + from.len();
        }
    }
    // FIELD-NAME swaps: a wrong struct field read/written (`first(&self) { self.b }` should be
    // `self.a`). For each `self.FIELD` whose FIELD is a known struct field, swap to each other field.
    let struct_fields = parse_struct_fields(code);
    if struct_fields.len() >= 2 {
        let names: Vec<&str> = struct_fields.iter().map(|(n, _)| n.as_str()).collect();
        let mut idx = 0;
        while let Some(pos) = code[idx..].find("self.") {
            let at = idx + pos + 5; // just past "self."
            let mut e = at;
            while e < code.len() && (cb[e].is_ascii_alphanumeric() || cb[e] == b'_') {
                e += 1;
            }
            let field = &code[at..e];
            if names.contains(&field) {
                for other in &names {
                    if *other != field {
                        out.push(splice_mutation(code, tail, at, field.len(), other));
                    }
                }
            }
            idx = (idx + pos + 5).max(e);
        }
    }
    // Integer literals -> +-1 (off-by-one).
    let mut i = 0;
    while i < code.len() {
        if cb[i].is_ascii_digit() && (i == 0 || !(cb[i - 1].is_ascii_alphanumeric() || cb[i - 1] == b'_')) {
            let s = i;
            while i < cb.len() && cb[i].is_ascii_digit() {
                i += 1;
            }
            // SOUNDNESS: never mutate a literal on the RHS of a `const`/`static` DECLARATION. Corrupting
            // a named constant (`const KIB = 1024` -> `1023`) to satisfy a weak test is a wrong "fix"
            // that silently breaks every other use of the constant — a real bug lives in a fn body, and
            // the legitimate wrong-constant class mutates literals in EXPRESSIONS, not declarations.
            let line_start = code[..s].rfind('\n').map(|p| p + 1).unwrap_or(0);
            let lead = code[line_start..s].trim_start();
            let in_const_decl = lead.starts_with("const ")
                || lead.starts_with("static ")
                || lead.starts_with("pub const ")
                || lead.starts_with("pub static ")
                || lead.starts_with("pub(crate) const ")
                || lead.starts_with("pub(crate) static ");
            if !in_const_decl {
                if let Ok(n) = code[s..i].parse::<i64>() {
                    for d in [n + 1, n - 1] {
                        out.push(splice_mutation(code, tail, s, i - s, &d.to_string()));
                    }
                }
            }
        } else {
            i += 1;
        }
    }
    // STRUCTURAL guard templates: insert an early-return edge guard at a function body's start —
    // `if p == 0 { return <edge>; }` / `if p.is_empty() { .. }` — with <edge> inferred from the
    // return type (None / 0 / false / empty). The classic missing-edge-case bug: `safe_div(a,b) ->
    // Some(a/b)` should guard `b == 0 -> None`. A single token swap can't express this; a template
    // can. Bounded and last-ish; cargo picks the guard/edge that actually passes.
    out.extend(structural_guard_mutations(code, tail));
    // BOOLEAN-RETURN negation: a `-> bool` fn whose predicate is inverted (`v.is_empty()` where
    // `!v.is_empty()` is meant, or a stray `!`). A single operator swap can't express a `!`; toggle
    // it on the return expression. Bounded (one per bool fn).
    out.extend(bool_return_negation_mutations(code, tail));
    // OPERAND function-wrap: a bare identifier operand `x` -> `F(x)` for each single-arg function F
    // defined in the code (the "wrong sub-expression" bug: `double(x) + x` -> `double(x) + double(x)`).
    // Last, so the cheaper operator/assignment/const fixes are tried first under the mutation cap.
    let fns: Vec<String> = single_arg_fn_names(code);
    if !fns.is_empty() {
        let mut i = 0;
        while i < code.len() {
            let is_ident_start = cb[i].is_ascii_alphabetic() || cb[i] == b'_';
            let boundary_before = i == 0 || !(cb[i - 1].is_ascii_alphanumeric() || cb[i - 1] == b'_');
            if is_ident_start && boundary_before {
                let s = i;
                while i < cb.len() && (cb[i].is_ascii_alphanumeric() || cb[i] == b'_') {
                    i += 1;
                }
                let ident = &code[s..i];
                // A VALUE operand: not a call (`x(`), not a field/method (`x.`/`.x`), not a keyword.
                let next = code[i..].chars().next();
                let after_dot = s > 0 && cb[s - 1] == b'.';
                // Skip TYPE positions/names: wrapping a type (`-> i64` -> `-> f(i64)`, `: Vec` ->
                // `: f(Vec)`) only makes non-compiling garbage that burns a cargo run. Cheap guards:
                // a primitive/known type name, an Uppercase-led ident (types/variants), or a type
                // annotation position (prev non-space char is `:`, or we sit just after `->`).
                let prev_ns = {
                    let mut j = s;
                    while j > 0 && (cb[j - 1] == b' ' || cb[j - 1] == b'\t') {
                        j -= 1;
                    }
                    (j > 0).then(|| cb[j - 1]).unwrap_or(0)
                };
                let after_arrow = prev_ns == b'>' && s >= 2 && code[..s].trim_end().ends_with("->");
                let is_type = prev_ns == b':'
                    || prev_ns == b'<'
                    || after_arrow
                    || ident.chars().next().is_some_and(|c| c.is_ascii_uppercase())
                    || matches!(ident, "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "u8" | "u16"
                        | "u32" | "u64" | "u128" | "usize" | "f32" | "f64" | "bool" | "char" | "str");
                let is_value = !matches!(next, Some('(') | Some('.') | Some('!'))
                    && !after_dot
                    && !is_type
                    && !matches!(ident, "let" | "mut" | "fn" | "if" | "else" | "return" | "while"
                        | "for" | "in" | "self" | "pub" | "struct" | "impl" | "true" | "false"
                        | "as" | "match" | "Some" | "None" | "Ok" | "Err");
                if is_value {
                    for f in &fns {
                        if f != ident {
                            out.push(splice_mutation(code, tail, s, ident.len(), &format!("{f}({ident})")));
                        }
                    }
                }
            } else {
                i += 1;
            }
        }
    }
    out
}

/// For each `-> bool` function, toggle a `!` on its return expression — the tail expression or a
/// `return EXPR;`. A common real bug is an inverted predicate (`v.is_empty()` vs `!v.is_empty()`),
/// which no single operator swap can express. Bounded: at most one candidate per bool function.
fn bool_return_negation_mutations(code: &str, tail: &str) -> Vec<String> {
    let mut out = Vec::new();
    let cb = code.as_bytes();
    let mut search = 0usize;
    while let Some(rel) = code[search..].find("-> bool") {
        let hdr = search + rel;
        search = hdr + 7;
        let Some(brel) = code[hdr..].find('{') else { continue };
        let bopen = hdr + brel;
        // brace-match the body
        let mut depth = 0i32;
        let mut close = None;
        let mut k = bopen;
        while k < cb.len() {
            match cb[k] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(k);
                        break;
                    }
                }
                _ => {}
            }
            k += 1;
        }
        let Some(close) = close else { continue };
        let inner_start = bopen + 1;
        let inner = &code[inner_start..close];
        // The return-expression byte span [es, ee): a `return EXPR;`, else the tail expression.
        let (es, ee) = if let Some(rp) = inner.rfind("return ") {
            let estart = inner_start + rp + 7;
            match code[estart..close].find(';') {
                Some(p) => (estart, estart + p),
                None => continue,
            }
        } else {
            let te = inner.trim_end();
            if te.is_empty() || te.ends_with(';') || te.ends_with('}') {
                continue; // a block/statement body, not a bare tail expression
            }
            let ee = inner_start + te.len();
            let es = match te.rfind(';') {
                Some(p) => inner_start + p + 1,
                None => inner_start,
            };
            (es, ee)
        };
        let raw = &code[es..ee];
        let expr = raw.trim();
        if expr.is_empty() || expr.contains('{') {
            continue; // skip `if { .. } else { .. }` and other block expressions
        }
        let toggled = match expr.strip_prefix('!') {
            Some(rest) => rest.trim().to_string(),
            None => format!("!({expr})"),
        };
        let lead = raw.len() - raw.trim_start().len();
        let es_t = es + lead;
        out.push(splice_mutation(code, tail, es_t, expr.len(), &toggled));
    }
    out
}

/// Insert an early-return edge guard at the start of each function body: `if <cond> { return
/// <edge>; }`, where `<edge>` is inferred from the return type and `<cond>` from each parameter
/// (`p == 0` for an int, `p.is_empty()` for a String/Vec). Covers the missing-edge-case bug class.
fn structural_guard_mutations(code: &str, tail: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut search = 0;
    while let Some(rel) = code[search..].find("fn ") {
        let fstart = search + rel;
        search = fstart + 3;
        // parse `fn NAME(params) -> RET {`
        let after = &code[fstart..];
        let Some(op) = after.find('(') else { continue };
        let ab = after.as_bytes();
        let mut depth = 0i32;
        let mut cp = None;
        for i in op..ab.len() {
            match ab[i] {
                b'(' => depth += 1,
                b')' => {
                    depth -= 1;
                    if depth == 0 {
                        cp = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let Some(cp) = cp else { continue };
        let params_str = &after[op + 1..cp];
        // find `->` RET and the body-open `{` after the params
        let post = &after[cp + 1..];
        let (ret, brace_rel) = match (post.find("->"), post.find('{')) {
            (Some(a), Some(bpos)) if a < bpos => (post[a + 2..bpos].trim().to_string(), bpos),
            (_, Some(bpos)) => (String::new(), bpos),
            _ => continue,
        };
        let body_open = fstart + cp + 1 + brace_rel + 1; // just after `{`
        let edges = guard_edge_returns(&ret);
        if edges.is_empty() {
            continue;
        }
        for (name, ty) in parse_typed_params(params_str) {
            if name == "self" {
                continue;
            }
            let conds: Vec<String> = if ty.contains("i64") {
                vec![format!("{name} == 0"), format!("{name} < 0")]
            } else if ty.contains("String") || ty.contains("Vec<") || ty.contains("[") {
                vec![format!("{name}.is_empty()")]
            } else {
                continue;
            };
            for cond in &conds {
                for edge in &edges {
                    let guard = format!(" if {cond} {{ return {edge}; }}");
                    let mut m = String::with_capacity(code.len() + tail.len() + guard.len() + 2);
                    m.push_str(&code[..body_open]);
                    m.push_str(&guard);
                    m.push_str(&code[body_open..]);
                    m.push_str(tail);
                    out.push(m);
                }
            }
        }
    }
    out
}

/// Candidate edge return expressions for a Rust return type.
fn guard_edge_returns(ret: &str) -> Vec<&'static str> {
    let r = ret.trim();
    if r.starts_with("Option<") {
        vec!["None"]
    } else if r == "bool" {
        vec!["false", "true"]
    } else if r == "i64" {
        vec!["0"]
    } else if r == "String" {
        vec!["String::new()"]
    } else if r.starts_with("Vec<") {
        vec!["Vec::new()"]
    } else {
        vec![]
    }
}

/// Parse `name: TYPE, ...` into (name, type) pairs (handles `&mut self`, `mut x`, generics via a
/// depth-aware comma split).
fn parse_typed_params(params: &str) -> Vec<(String, String)> {
    let mut parts: Vec<String> = Vec::new();
    let mut depth = 0i32;
    let mut last = 0;
    for (i, c) in params.char_indices() {
        match c {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(params[last..i].to_string());
                last = i + 1;
            }
            _ => {}
        }
    }
    parts.push(params[last..].to_string());
    parts
        .into_iter()
        .filter_map(|p| {
            let p = p.trim();
            if p.is_empty() {
                return None;
            }
            let (nm, ty) = p.split_once(':')?;
            Some((nm.trim().trim_start_matches("mut ").trim().to_string(), ty.trim().to_string()))
        })
        .collect()
}

/// Names of functions defined in `code` that take exactly one parameter — candidates for wrapping a
/// bare operand as `F(operand)`.
fn single_arg_fn_names(code: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut rest = code;
    while let Some(p) = rest.find("fn ") {
        let after = &rest[p + 3..];
        let name: String = after.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
        if let Some(op) = after.find('(') {
            if let Some(cp) = after[op..].find(')') {
                let params = &after[op + 1..op + cp];
                let n = params.split(',').filter(|s| !s.trim().is_empty()).count();
                if n == 1 && !name.is_empty() && !names.contains(&name) {
                    names.push(name);
                }
            }
        }
        rest = &after;
    }
    names
}

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

    // Give the model the CODE + the concrete failure and let IT localize — a real coding-agent edit.
    // A pre-localized single-fn rewrite picks the WRONG function when the query is generic and the
    // failing test exercises several (e.g. a struct with `new` + `add`, where only `add` is buggy):
    // the model must SEE the code and the failure to know which function to fix. It returns the
    // corrected function(s), which are then applied by NAME to whichever repo file defines them.
    let failure = analysis
        .map(|a| format!("\n\nThe failure: {} ({})", a.message, a.suggested_action))
        .unwrap_or_default();
    let request = format!(
        "A test is failing. Here is the code:\n```rust\n{old_text}\n```{failure}\n\n\
         Task: {description}\n\nOutput ONLY the single corrected Rust function (its full signature \
         and body, e.g. `pub fn ...`) that makes the failing test pass. Change nothing else."
    );

    // Turn a model response (Rust source string) into a candidate patch, EXACTLY as before: prefer a
    // multi-file swap (>= 2 fns the repo already defines), else a single-fn body-swap applied by the
    // fn's OWN name to whatever file defines it (falling back to the pre-resolved repo_fn / picked
    // target). Pure over the response + context, so it runs once per attempt with no cargo cost.
    let build_patch = |program: &str| -> Option<RepairPatch> {
        if let Some(patch) = model_response_to_multifile_patch(context, program) {
            return Some(patch);
        }
        let model_fn = first_fn_name_in_source(program).unwrap_or_else(|| repo_fn.clone());
        let (tgt, tgt_text) = context
            .files
            .iter()
            .filter(|f| f.path.ends_with(".rs"))
            .find_map(|f| {
                let t = f.text.as_deref()?;
                file_defines_function(t, &model_fn).then(|| (f.path.clone(), t.to_string()))
            })
            .unwrap_or_else(|| (target.clone(), old_text.clone()));
        let new_text = model_body_to_new_text(&tgt_text, &model_fn, program)?;
        Some(
            RepairPatch::new()
                .with_edit(RepairEdit::new(
                    tgt,
                    tgt_text,
                    new_text,
                    "gated model-repair proposer (untrusted; cargo-test gated)",
                ))
                .with_metadata("proposer", "model_repair"),
        )
    };

    // A COMPILE-ERROR REPAIR LOOP over the model, mirroring try_mutation_repair_patch's
    // write -> cargo-verify -> revert scheme: apply each candidate patch to the work copy, run the
    // task's test command, and return the patch ONLY when cargo went green (the RepairLoop re-applies
    // the winner). A failing candidate's stderr is fed BACK to the model as its prior attempt+error
    // (propose_rust_fn's `prior` hook) so retries actually differ. This recovers the model's
    // logically-correct-but-non-compiling Rust while preserving never-wrong: we ship nothing that did
    // not make cargo pass on the work copy.
    let verifier = RepairVerifier::new(&context.root, GuardrailPolicy::default());
    // Apply every whole-file edit of `patch` to the work copy; return (abs_path, original_text) so the
    // caller can revert. Skips edits that fail to write.
    let apply = |patch: &RepairPatch| -> Vec<(std::path::PathBuf, String)> {
        let mut undo = Vec::new();
        for e in &patch.edits {
            let abs = std::path::Path::new(&context.root).join(&e.path);
            if std::fs::write(&abs, &e.new_text).is_ok() {
                undo.push((abs, e.old_text.clone()));
            }
        }
        undo
    };
    let revert = |undo: Vec<(std::path::PathBuf, String)>| {
        for (abs, orig) in undo {
            let _ = std::fs::write(&abs, &orig);
        }
    };

    let mut prior_code: Option<String> = None;
    let mut prior_err: Option<String> = None;
    for _ in 0..3 {
        let prior = match (&prior_code, &prior_err) {
            (Some(c), Some(e)) => Some((c.as_str(), e.as_str())),
            _ => None,
        };
        // Ask for RUST, not Mog: the repo-repair oracle is cargo test over real Rust, and small local
        // models write Rust well but not the Mog DSL. Feed the prior attempt+error on retries.
        let Some(program) = crate::local_llm::propose_rust_fn(&request, prior, 0.2) else {
            break;
        };
        let Some(patch) = build_patch(&program) else {
            // Couldn't localize this response to any repo file — retry with a nudge.
            prior_code = Some(program);
            prior_err = Some("your function did not match any function defined in the repo".to_string());
            continue;
        };
        let undo = apply(&patch);
        let fully_applied = undo.len() == patch.edits.len();
        let v = verifier.verify(&task.test_command);
        revert(undo); // ALWAYS revert; the outer RepairLoop re-applies the winner.
        match v {
            // Return ONLY a patch that FULLY applied AND made cargo pass — otherwise a partial-write
            // patch could be verified against a subset yet returned whole (never-wrong guard).
            Ok(ver) if ver.success && fully_applied => return Some(patch),
            Ok(ver) => {
                prior_code = Some(program);
                // Feed BOTH stdout+stderr back: rustc COMPILE errors land on stderr, but cargo-test
                // ASSERTION diffs (expected vs actual) land on STDOUT — the model needs both to fix
                // its own logic, not just "test failed".
                let mut err = ver.failure_output();
                if err.trim().is_empty() {
                    err = "the tests still fail".to_string();
                }
                if err.len() > 1500 {
                    let end = (0..=1500).rev().find(|&i| err.is_char_boundary(i)).unwrap_or(0);
                    err.truncate(end);
                }
                prior_err = Some(err);
            }
            Err(_) => {
                prior_code = Some(program);
                prior_err = Some("verification could not run".to_string());
            }
        }
    }
    // No candidate made cargo pass — refuse rather than ship unverified.
    None
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
        // VERBATIM FIRST: a full `pub fn NAME` whose signature already matches the repo fn is
        // applied byte-for-byte (correct Rust is never mangled/dropped by the reshape heuristics).
        if let Some(next) = verbatim_repo_fn_replacement(&entry.1, name, src) {
            entry.1 = next;
            continue;
        }
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
                "gated model multi-function repair (untrusted; cargo-test gated)",
            ));
            files_changed += 1;
        }
    }
    // We already required >= 2 proposed functions (top of fn); accept the patch whether they land in
    // one file (the common case: a struct whose methods span a single lib.rs -- add/count/median) or
    // several. This is what completes a MIXED crate the templates can't fully fill: the model returns
    // ALL the still-empty functions and they're applied together, then cargo gates the whole thing.
    if files_changed >= 1 {
        Some(patch.with_metadata("proposer", "model_repair_multifn"))
    } else {
        None
    }
}

/// Pure string core of the model-repair stage (TESTABLE without a model or repo):
/// reshape a proposed program to `repo_fn`'s signature in `old_text`. Declines when
/// the proposal is not plain compilable Rust (IR wrappers / unlowered `:=`) or is a
/// no-op — the same gate the verified synthesis path applies.
fn model_body_to_new_text(old_text: &str, repo_fn: &str, program: &str) -> Option<String> {
    // VERBATIM FIRST: a signature-matching full fn is applied as-is (never lost to reshape).
    if let Some(next) = verbatim_repo_fn_replacement(old_text, repo_fn, program) {
        if next != *old_text {
            return Some(next);
        }
    }
    let body = rust_code_for_repo_synthesis(program);
    if !is_plain_rust_body(&body) {
        return None;
    }
    let new_text = reshape_to_repo_signature(old_text, repo_fn, &body)?;
    (new_text != *old_text).then_some(new_text)
}

// ── GATED MODEL-INTENT stage: model proposes a SPEC, the engine + oracle DISPOSE ──
//
// A second, distinct model lane from `try_model_repair_patch` (which asks the model
// for a Rust BODY). Here the model proposes ONLY I/O examples (the WHAT); the
// DETERMINISTIC ENGINE synthesizes the program (the HOW) and the repo cargo-test
// oracle DISPOSES. On acceptance the engine-synthesized program is DISTILLED so the
// capability is absorbed model-free — the model teaches once. The model never emits
// code that reaches the repo unchecked, and the guarantee never depends on it.

/// Staged model-INTENT distillation candidates, keyed by task id. A model-intent
/// patch stashes its ENGINE-VERIFIED `(problem, code)` here; the supervisor consumes
/// it (via [`distill_accepted_model_solve`]) ONLY after the repo cargo-test oracle
/// ACCEPTS the patch — so a model spec that fails the repo test never teaches the
/// store. Process-global + keyed by task id so it survives the closure boundary the
/// repair loop crosses when it applies + re-verifies a patch.
#[allow(clippy::type_complexity)]
fn distillation_stage(
) -> &'static std::sync::Mutex<std::collections::HashMap<String, (crate::benchmark::Problem, String)>>
{
    static STAGE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<String, (crate::benchmark::Problem, String)>>,
    > = std::sync::OnceLock::new();
    STAGE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Stash an engine-verified `(problem, code)` for `task_id` awaiting repo-test accept.
pub(crate) fn stage_model_distillation(
    task_id: &str,
    problem: crate::benchmark::Problem,
    code: String,
) {
    if let Ok(mut g) = distillation_stage().lock() {
        g.insert(task_id.to_string(), (problem, code));
    }
}

/// Remove and return a staged candidate for `task_id` (does NOT record it).
pub fn take_model_distillation(task_id: &str) -> Option<(crate::benchmark::Problem, String)> {
    distillation_stage().lock().ok()?.remove(task_id)
}

/// Consume a staged model-INTENT distillation candidate for `task_id` and, IF one
/// exists, DISTILL it via [`crate::op_library::record_proposed_op`] so a FUTURE run
/// solves the same task MODEL-FREE (at the library/learned tier, before the model is
/// ever consulted). Call ONLY after the repo cargo-test oracle ACCEPTED the patch.
///
/// Inert (returns `false`) when nothing was staged (no model in the loop — the common
/// case with `NSYNTH_LOCAL_LLM_URL` unset) or the learned-store path is unset
/// (`record_proposed_op` no-ops without `NSYNTH_LEARNED_OPS_PATH`). So this is a
/// zero-effect call on any default run.
pub fn distill_accepted_model_solve(task_id: &str) -> bool {
    match take_model_distillation(task_id) {
        Some((problem, code)) => crate::op_library::record_proposed_op(&problem, &code),
        None => false,
    }
}

/// Drop any staged candidate for `task_id` WITHOUT distilling — a run that did not end
/// in a repo-test accept must never teach the store.
pub fn discard_model_distillation(task_id: &str) {
    let _ = take_model_distillation(task_id);
}

/// Map a raw JSON value from a model spec to a runtime `Value` (int / bool / string /
/// int-array / heterogeneous array). `None` for anything outside the verified domains.
fn model_json_to_value(v: &serde_json::Value) -> Option<crate::benchmark::Value> {
    use crate::benchmark::Value;
    if let Some(b) = v.as_bool() {
        return Some(Value::Bool(b));
    }
    if let Some(i) = v.as_i64() {
        return Some(Value::Int(i));
    }
    if let Some(s) = v.as_str() {
        return Some(Value::Str(s.to_string()));
    }
    if let Some(arr) = v.as_array() {
        if let Some(ints) = arr.iter().map(|x| x.as_i64()).collect::<Option<Vec<i64>>>() {
            return Some(Value::int_array(&ints));
        }
        let vals: Option<Vec<Value>> = arr.iter().map(model_json_to_value).collect();
        return Some(Value::Array(vals?));
    }
    None
}

/// GATED MODEL-INTENT stage — the untrusted SPEC proposer of last resort.
///
/// When every deterministic path has declined, an OPTIONAL local LLM proposes I/O
/// EXAMPLES for the task; the engine then synthesizes + verifies deterministically.
/// It is a PROPOSER OF INTENT ONLY — the model's examples are UNTRUSTED and the
/// engine's synthesized program is reshaped to the repo fn's exact signature and
/// handed back as an ordinary patch, so the caller's cargo-test acceptance oracle
/// still decides.
///
/// Inert by default: with `NSYNTH_LOCAL_LLM_URL` unset the lane returns `None`
/// immediately (and `propose_examples` is likewise off), so there is zero behaviour
/// change on any machine without a model.
pub fn try_model_intent_patch(
    task: &RepoTaskSpec,
    context: &RepairContext,
    description: &str,
) -> Option<RepairPatch> {
    // Off unless an endpoint is configured — the guarantee is the engine + cargo-test
    // oracle, never the model. Checked here too (not only inside `propose_examples`)
    // so the lane is provably inert without reaching the network layer.
    if std::env::var("NSYNTH_LOCAL_LLM_URL")
        .ok()
        .filter(|u| !u.trim().is_empty())
        .is_none()
    {
        return None;
    }
    let request = format!(
        "{description}\n\nGive correct input/output examples for this function \
         (integers, arrays of integers, or booleans)."
    );
    let proposed = crate::local_llm::propose_examples(&request)?;
    let rows: Vec<crate::benchmark::Example> = proposed
        .iter()
        .filter_map(|p| {
            let inputs: Vec<crate::benchmark::Value> =
                p.inputs.iter().map(model_json_to_value).collect::<Option<_>>()?;
            let expected = model_json_to_value(&p.output)?;
            Some(crate::benchmark::Example { inputs, expected })
        })
        .collect();
    try_model_intent_patch_from_spec(task, context, description, rows)
}

/// Deterministic core of the model-INTENT stage: given a model-proposed SPEC (`rows`,
/// UNTRUSTED I/O examples), synthesize + strictly verify a program and, on success,
/// reshape it onto the repo fn's signature + STAGE it for distillation. Factored out
/// so it is testable with NO server — a test passes a hand-crafted spec (a stub for
/// the model's output) and asserts a deliberately-WRONG spec is REJECTED (never
/// promoted to a patch, nothing staged) while a correct spec yields a verified patch.
pub(crate) fn try_model_intent_patch_from_spec(
    task: &RepoTaskSpec,
    context: &RepairContext,
    description: &str,
    mut rows: Vec<crate::benchmark::Example>,
) -> Option<RepairPatch> {
    use crate::benchmark::{Problem, Value};

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

    rows.sort_by(|a, b| (&a.inputs, &a.expected).cmp(&(&b.inputs, &b.expected)));
    rows.dedup();
    // Need a genuine HELD-OUT split: enough points that the seed pins the function
    // AND at least one example is withheld from synthesis to catch a fit-to-seed spec.
    if rows.len() < 4 {
        return None;
    }
    // Array-RETURNING functions need a Vec/slice reshape this proposer does not do;
    // decline them (mirrors `try_test_mined_synthesis_patch`). Scalar and boolean
    // outputs — incl. array->scalar folds — proceed.
    if rows.iter().any(|e| matches!(e.expected, Value::Array(_))) {
        return None;
    }

    // HELD-OUT split: synthesize from all-but-last, judge on ALL incl. the held-out.
    let seed = &rows[..rows.len() - 1];
    let sig: &'static str =
        Box::leak(crate::linguigenesis_bridge::infer_signature(&repo_fn, &rows).into_boxed_str());
    let seed_problem = Problem {
        name: repo_fn.clone(),
        category: "repo-model-intent",
        description: "",
        signature: sig,
        examples: seed.to_vec(),
        ..Default::default()
    };
    let res = crate::solver::solve_problem(&seed_problem);
    if !res.success {
        return None;
    }
    // The engine's program must reproduce EVERY example incl. the genuinely held-out
    // one — the generalization oracle a fit-to-seed model spec misses.
    if !crate::runtime::code_reproduces_examples(&res.code, &rows) {
        return None;
    }
    // Robustness floor: clean execution on perturbations of the examples.
    let all_problem = Problem {
        name: repo_fn.clone(),
        category: "repo-model-intent",
        description: "",
        signature: sig,
        examples: rows.clone(),
        ..Default::default()
    };
    if crate::runtime::verify_problem_code_strict(&all_problem, &res.code).is_err() {
        return None;
    }
    // Never adopt a magic-constant memorization fit.
    if crate::synth_confidence::is_memorization_overfit(&res.code, &rows) {
        return None;
    }
    let synthesized = rust_code_for_repo_synthesis(&res.code);
    if !is_plain_rust_body(&synthesized) {
        return None;
    }
    let new_text = reshape_to_repo_signature(&old_text, &repo_fn, &synthesized)?;
    if new_text == old_text {
        return None;
    }

    // STAGE the ENGINE-VERIFIED (problem, program) for distillation — the supervisor
    // consumes it ONLY if the repo cargo-test oracle accepts this patch. Model teaches
    // once, and only on a TRUE accept.
    stage_model_distillation(&task.id, all_problem, res.code.clone());

    Some(
        RepairPatch::new()
            .with_edit(RepairEdit::new(
                target,
                old_text,
                new_text,
                "gated model-INTENT proposer (model spec -> engine-synthesized, verified; cargo-test gated)",
            ))
            .with_metadata("proposer", "model_intent")
            .with_metadata("synthesis_method", res.method.clone()),
    )
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
    analysis: Option<&FailureAnalysis>,
) -> Option<RepairPatch> {
    let intent = CodingIntent::from_nl_lenient(description).ok();
    let target = pick_target_path(task, context, intent.as_ref()).ok()?;
    let old_text = read_relative_file(context, &target).ok()?;
    let default_fn = intent
        .as_ref()
        .map(|i| i.function_name.strip_prefix("nl_").unwrap_or(&i.function_name).to_string())
        .unwrap_or_default();
    let resolved_fn = resolve_repo_fn_name(&default_fn, Some(&old_text));

    // TARGET FUNCTION, in PRIORITY order.
    //  1. the function named by the CURRENT failing assert (from `analysis` file:line). This is what
    //     makes MULTI-FUNCTION crates converge: once an iteration repairs fn A, cargo's next failure
    //     points at fn B's assert, so the next iteration targets B instead of re-repairing A. Without
    //     this the description resolves to the SAME fn every iteration and a second broken fn is never
    //     reached (the loop stalls / exhausts iterations).
    //  2. the description-resolved name — the fallback, and for a single-function crate it coincides
    //     with (1) so behavior is identical.
    // Each candidate is mined independently; `synthesize_mined_for_fn` returns None (skipped) when
    // the fn is already correct or its examples don't verify, so a stale/wrong candidate can't emit a
    // bad patch — and the real cargo test still gates every result (never-wrong preserved).
    let mut candidates: Vec<String> = Vec::new();
    if let Some(f) = asserted_fn_at_failure(context, analysis) {
        candidates.push(f);
    }
    if !candidates.iter().any(|c| c == &resolved_fn) {
        candidates.push(resolved_fn.clone());
    }

    for repo_fn in &candidates {
        let mut rows: Vec<(Vec<crate::benchmark::Value>, crate::benchmark::Value)> = Vec::new();
        for f in &context.files {
            if let Some(t) = f.text.as_deref() {
                rows.extend(mine_asserts(t, repo_fn));
            }
        }
        rows.sort();
        rows.dedup();
        if rows.len() < 2 {
            continue;
        }
        if let Some(patch) =
            synthesize_mined_for_fn(context, description, repo_fn, rows, &target, &old_text)
        {
            return Some(patch);
        }
    }

    // NAME-RECOVERY FALLBACK. None of the candidates pinned ≥2 rows: the NL intent mis-resolved the
    // name — "count how many elements are positive" resolves to `is_positive` (the "positive" cue),
    // whose asserts don't exist — or the function is MISSING entirely, so there is no definition for
    // `resolve_repo_fn_name` to anchor on. The failing asserts NAME the function they call
    // (`assert_eq!(count_positives(..), 3)`), so recover the dominant asserted call-name and re-mine.
    // Purely additive: runs only after every candidate failed to pin rows; the recovered candidate is
    // still strict-verified + cargo-gated downstream.
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for f in &context.files {
        if let Some(t) = f.text.as_deref() {
            accumulate_asserted_call_names(t, &mut counts);
        }
    }
    let recovered = counts
        .into_iter()
        .filter(|(n, _)| !candidates.iter().any(|c| c == n))
        .max_by_key(|(_, c)| *c)
        .map(|(n, _)| n);
    let name = recovered?;
    let mut rows: Vec<(Vec<crate::benchmark::Value>, crate::benchmark::Value)> = Vec::new();
    for f in &context.files {
        if let Some(t) = f.text.as_deref() {
            rows.extend(mine_asserts(t, &name));
        }
    }
    rows.sort();
    rows.dedup();
    if rows.len() < 2 {
        return None;
    }
    synthesize_mined_for_fn(context, description, &name, rows, &target, &old_text)
}

/// The function named by the CURRENT failing assert, recovered from `analysis`'s `file:line`. Reads
/// the implicated file at the failing line and returns the dominant `assert_eq!` call-name in a small
/// window from there (covers a multi-line assert). `None` when there is no analysis, no line, the file
/// isn't in context, or the failing line is not an assert (e.g. a compile error with no call) — the
/// caller then falls back to the description-resolved name.
fn asserted_fn_at_failure(
    context: &RepairContext,
    analysis: Option<&FailureAnalysis>,
) -> Option<String> {
    let window = failing_assert_window(context, analysis)?;
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    accumulate_asserted_call_names(&window, &mut counts);
    counts.into_iter().max_by_key(|(_, c)| *c).map(|(n, _)| n)
}

/// The small source window around the CURRENT failing assert (`analysis` file:line): the failing line
/// plus a couple after it, to cover a multi-line `assert_eq!`. `None` when there is no analysis/line or
/// the implicated file isn't in context. Both the single-fn localizer and the multi-function repair
/// read the functions the FAILING TEST names from this window — the repo's own test decides which
/// function(s) to fix, never the prose.
fn failing_assert_window(
    context: &RepairContext,
    analysis: Option<&FailureAnalysis>,
) -> Option<String> {
    let a = analysis?;
    let file = a.file.as_deref()?;
    let line = a.line? as usize; // 1-based
    if line == 0 {
        return None;
    }
    let text = context
        .files
        .iter()
        .find(|f| f.path == file || f.path.ends_with(file) || file.ends_with(f.path.as_str()))
        .and_then(|f| f.text.as_deref())?;
    let lines: Vec<&str> = text.lines().collect();
    if line > lines.len() {
        return None;
    }
    let start = line - 1;
    let end = (start + 3).min(lines.len());
    Some(lines[start..end].join("\n"))
}

/// COMPOUND / MULTI-FUNCTION repair, grounded in the REPO — not the prose. A single failing assert can
/// name TWO OR MORE distinct functions (`assert_eq!(add(2, 3) + sub(9, 4), 10)`); the per-iteration
/// single-function localizer can only fix one and the assert keeps failing (the loop stalls). This
/// stage fixes ALL the functions the failing assert (or, failing that, the whole crate's asserts) names,
/// in ONE atomic patch. The set of functions comes entirely from the asserted call-names
/// (`accumulate_asserted_call_names`) — the repo's own tests decide "how many functions", never a regex
/// on "and". Each function is synthesized + strict-verified independently and the fixes are CHAINED onto
/// the file's text, so the single emitted edit is behavior-checked as a unit by the real cargo test.
/// Returns None (single-function path handles it) unless at least TWO functions are actually repaired,
/// so it never fires — nor over-splits — on an ordinary single-function crate.
pub fn try_multifn_mined_patch(
    task: &RepoTaskSpec,
    context: &RepairContext,
    description: &str,
    analysis: Option<&FailureAnalysis>,
) -> Option<RepairPatch> {
    let intent = CodingIntent::from_nl_lenient(description).ok();
    let target = pick_target_path(task, context, intent.as_ref()).ok()?;
    let original = read_relative_file(context, &target).ok()?;

    // Candidate functions come from the REPO's asserts: the distinct call-names in the FAILING assert
    // first (most precise — exactly the functions this failure implicates), else every asserted
    // call-name in the crate. This is the emergent "how many functions" signal; prose is never parsed.
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    if let Some(window) = failing_assert_window(context, analysis) {
        accumulate_asserted_call_names(&window, &mut counts);
    }
    if counts.len() < 2 {
        counts.clear();
        for f in &context.files {
            if let Some(t) = f.text.as_deref() {
                accumulate_asserted_call_names(t, &mut counts);
            }
        }
    }
    // Only functions DEFINED in the target file can be chained into one single-file edit here; a
    // function that lives elsewhere is left to the per-iteration single-fn path (which retargets it).
    let mut candidates: Vec<String> = counts
        .into_keys()
        .filter(|n| original.contains(&format!("fn {n}(")) || original.contains(&format!("fn {n} ")))
        .collect();
    candidates.sort(); // deterministic
    if candidates.len() < 2 {
        return None;
    }

    // Chain each function's verified fix onto the accumulating file text. `synthesize_mined_for_fn`
    // reshapes onto the text we pass as `picked_old_text`, and does NOT retarget when the function is
    // defined there (it is — we filtered to target-defined fns), so each fix composes on the previous.
    let mut acc = original.clone();
    let mut fixed = 0usize;
    for repo_fn in &candidates {
        let mut rows: Vec<(Vec<crate::benchmark::Value>, crate::benchmark::Value)> = Vec::new();
        for f in &context.files {
            if let Some(t) = f.text.as_deref() {
                rows.extend(mine_asserts(t, repo_fn));
            }
        }
        rows.sort();
        rows.dedup();
        if rows.len() < 2 {
            continue;
        }
        if let Some(patch) =
            synthesize_mined_for_fn(context, description, repo_fn, rows, &target, &acc)
        {
            if let Some(edit) = patch.edits.iter().find(|e| e.path == target) {
                if edit.new_text != acc {
                    acc = edit.new_text.clone();
                    fixed += 1;
                }
            }
        }
    }
    if fixed < 2 || acc == original {
        return None;
    }
    Some(
        RepairPatch::new()
            .with_edit(RepairEdit::new(
                target,
                original,
                acc,
                "multi-function test-mined synthesis (each fn verified; whole patch cargo-gated)",
            ))
            .with_metadata("proposer", "nl_multifn_test_mined")
            .with_metadata("functions_fixed", fixed.to_string()),
    )
}

/// Synthesize one verified repair for `repo_fn` from its mined I/O `rows`. Returns `None` when the
/// examples don't resolve/verify OR when the result would not change the file (the function is
/// already correct) — the latter lets the multi-function caller skip past already-fixed functions.
/// `picked_target`/`picked_old_text` are pick_target_path's choice, used as the feature-add append
/// site when no context file defines `repo_fn`.
fn synthesize_mined_for_fn(
    context: &RepairContext,
    description: &str,
    repo_fn: &str,
    rows: Vec<(Vec<crate::benchmark::Value>, crate::benchmark::Value)>,
    picked_target: &str,
    picked_old_text: &str,
) -> Option<RepairPatch> {
    let mut target = picked_target.to_string();
    let mut old_text = picked_old_text.to_string();
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
        .find(|f| f.text.as_deref().map(|t| file_defines_function(t, repo_fn)).unwrap_or(false))
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
    // Array-RETURNING functions (reverse, sort, k-largest) now proceed: `gencode_normalize`
    // fixes the Vec-return lowering (`= []` etc.) and `reshape_to_repo_signature` handles the
    // slice/by-value bridge. If a particular array signature still can't be reshaped, reshape
    // returns None and the proposer declines gracefully (same fall-through as before) — never a
    // bad patch, since strict-verify + the real cargo test still gate every candidate.
    let sig: &'static str =
        Box::leak(crate::linguigenesis_bridge::infer_signature(&repo_fn, &exs).into_boxed_str());
    let problem = crate::benchmark::Problem {
        name: repo_fn.to_string(),
        category: "repo-test-mined",
        description: "",
        signature: sig,
        examples: exs.clone(),
        ..Default::default()
    };
    // FRONT DOOR FIRST, then raw synthesis. The never-wrong router (`answer`) reaches the
    // ~300-op verified LIBRARY the bare enumerator cannot re-derive — "count how many
    // elements are positive" IS the library op `count_positives` (a filter+count), which the
    // raw solver refuses to synthesize from the same 2-3 mined points. `answer` proposes an op
    // from the prose AND verifies it against the mined examples, so it only returns a
    // library/composition/model op that reproduces the failing test's asserts; a WRONG guess
    // is refused by that gate (and the real cargo test is still the final oracle). Falls back
    // to `solve_problem` when the prose names no reachable op.
    let (code, method) = front_door_mined_code(description, &exs)
        .or_else(|| {
            let res = crate::solver::solve_problem(&problem);
            (res.success && crate::runtime::verify_problem_code_strict(&problem, &res.code).is_ok())
                .then(|| (res.code, res.method))
        })?;
    // NAME CANONICALIZATION. A front-door LIBRARY op keeps its OWN canonical name — "double the
    // input" resolves to `times_two`, "the sum of a list" to `array_sum` — but `problem.name`
    // (and the repo patch) address `repo_fn`. The strict re-verify below runs a `main` that calls
    // `problem.name`, so a differently-named single op would spuriously fail here even though the
    // router already verified it reproduces every mined example. Rename the op's entry fn to
    // `repo_fn` so the verify addresses the right symbol; `reshape_to_repo_signature` renames to
    // the SAME target downstream, so this is consistent. Only single-function ops are renamed —
    // a multi-fn composition's entry is selected structurally inside reshape, not by first-fn here,
    // so those are left untouched (unchanged behavior; they simply don't gain this rescue).
    let code = match first_fn_name_in_source(&code) {
        Some(entry) if entry != repo_fn && split_top_level_functions(&code).len() == 1 => {
            rename_first_fn(&code, repo_fn)
        }
        _ => code,
    };
    if crate::runtime::verify_problem_code_strict(&problem, &code).is_err() {
        return None;
    }
    // Same memorization guard as the CLI/teach paths: never adopt a magic-constant fit.
    if crate::synth_confidence::is_memorization_overfit(&code, &exs) {
        return None;
    }
    let res_method = method;
    let synthesized = rust_code_for_repo_synthesis(&code);
    if !is_plain_rust_body(&synthesized) {
        return None;
    }
    let new_text = if repo_has_fn {
        reshape_to_repo_signature(&old_text, repo_fn, &synthesized)?
    } else {
        // MISSING function (feature-add): reshape yields the standalone fn source; APPEND it to
        // the file rather than replacing the file contents.
        let new_fn = reshape_to_repo_signature(&old_text, repo_fn, &synthesized)?;
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
            .with_metadata("synthesis_method", res_method.clone()),
    )
}

/// Resolve mined I/O examples through the never-wrong front door (`verified_nl_router::answer`),
/// which proposes a verified LIBRARY op / composition / gated-model program from the prose and
/// gates it against the SAME examples. Returns `(mog_code, method)` only for a candidate that
/// reproduces every mined example — reaching the op library (e.g. `count_positives`) the raw
/// enumerator cannot re-derive — or `None` when the router refuses. The caller re-runs
/// `verify_problem_code_strict` and the real cargo test still gates the patch.
fn front_door_mined_code(
    description: &str,
    exs: &[crate::benchmark::Example],
) -> Option<(String, String)> {
    use crate::verified_nl_router::Answer;
    match crate::verified_nl_router::answer(description, exs) {
        Answer::Library { name, code } => Some((code, format!("front-door:library:{name}"))),
        Answer::Composition { code } => Some((code, "front-door:composition".to_string())),
        Answer::Synthesized { method, code } => Some((code, format!("front-door:{method}"))),
        Answer::Proposed { method, code } => Some((code, format!("front-door:proposed:{method}"))),
        Answer::Refused => None,
    }
}

/// Mine I/O examples for `fn_name` from `assert_eq!(..)` calls in `text`. Both
/// `assert_eq!(f(4), 8)` and `assert_eq!(8, f(4))` yield `([4], 8)`. Integer, string
/// (`"abc"`), and boolean (`true`/`false`) literals are captured — the solver's
/// verified domains; anything else (floats, expressions, method chains) is skipped.
/// The shared "does this synthesized program actually satisfy the failing test?" gate. Mines the
/// failing test's `assert_eq!` I/O for `repo_fn` and returns true iff the candidate reproduces
/// EVERY one (or there are none to check). Each PROSE-GROUNDED synthesis stage (emergent / real)
/// calls this before returning: it grounds via the low-confidence bridge on the intent's own
/// examples — nothing about THIS repo's failing test — so without the gate a mis-grounding
/// (count_positives->reverse-digits, mode->`last`, prefix-sums->`array_sum`) returns a wrong patch
/// that, running early in the ladder, short-circuits the example-verified stages that would solve
/// correctly (and makes the ladder flaky). Fail-closed on a mis-grounding; no asserts -> permissive.
fn synthesis_reproduces_failing_asserts(
    context: &RepairContext,
    repo_fn: &str,
    mog_code: &str,
) -> bool {
    let mut rows = Vec::new();
    for f in &context.files {
        if let Some(t) = f.text.as_deref() {
            rows.extend(mine_asserts(t, repo_fn));
        }
    }
    rows.sort();
    rows.dedup();
    if rows.is_empty() {
        return true;
    }
    let exs: Vec<crate::benchmark::Example> = rows
        .into_iter()
        .map(|(inputs, expected)| crate::benchmark::Example { inputs, expected })
        .collect();
    crate::runtime::code_reproduces_examples(mog_code, &exs)
}

/// Name matching is word-boundary safe, so `add` never matches `add_two`.
pub(crate) fn mine_asserts(text: &str, fn_name: &str) -> Vec<(Vec<crate::benchmark::Value>, crate::benchmark::Value)> {
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

/// Accumulate, into `counts`, the name of the function each `assert_eq!` in `text` CALLS. For
/// `assert_eq!(count_positives(vec![..]), 3)` the call side is `count_positives(vec![..])` (the
/// argument that is NOT a bare literal) and the recorded name is `count_positives`. Lets the
/// test-mined stage recover the function the test actually pins when the NL intent guessed a
/// different (or missing) name. Frequency-counted so the dominant asserted call wins.
fn accumulate_asserted_call_names(text: &str, counts: &mut std::collections::HashMap<String, usize>) {
    let mut cur = text;
    while let Some(rel) = cur.find("assert_eq!") {
        let after = &cur[rel + "assert_eq!".len()..];
        cur = after;
        let Some(open) = after.find('(') else { continue };
        let Some(inner) = balanced_parens(&after[open..]) else { continue };
        for arg in split_top_level_comma(inner) {
            let arg = arg.trim();
            // The call side is the argument that is NOT itself a bare literal.
            if parse_literal(arg).is_some() {
                continue;
            }
            if let Some(name) = leading_call_ident(arg) {
                *counts.entry(name).or_default() += 1;
            }
        }
    }
}

/// The identifier that begins a `NAME( … )` call at the start of `s` (after leading whitespace and
/// an optional `&`/`*`), or `None` when `s` does not start with a call. Rust-identifier chars only.
fn leading_call_ident(s: &str) -> Option<String> {
    let s = s.trim_start().trim_start_matches(['&', '*', ' ']);
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
        i += 1;
    }
    if i == 0 || bytes[0].is_ascii_digit() {
        return None;
    }
    // Must actually be a call: the next non-space character is `(`.
    if !s[i..].trim_start().starts_with('(') {
        return None;
    }
    Some(s[..i].to_string())
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
    // A bare `"..."` OR a String constructor around one: `"x".to_string()`, `"x".to_owned()`,
    // `"x".into()`, `String::from("x")`. Repo fns often take `String`, so their asserts wrap it.
    let s = t
        .strip_suffix(".to_string()")
        .or_else(|| t.strip_suffix(".to_owned()"))
        .or_else(|| t.strip_suffix(".into()"))
        .unwrap_or(t);
    let s = s.trim();
    let s = s
        .strip_prefix("String::from(")
        .and_then(|inner| inner.strip_suffix(')'))
        .map(str::trim)
        .unwrap_or(s);
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        let body = &s[1..s.len() - 1];
        if !body.contains('\\') {
            return Some(Value::Str(body.to_string()));
        }
    }
    // Int-array literal: `vec![1, 2, 3]` or `[1, 2, 3]` — the array domain the solver and op
    // library reason over. Needed to mine array-shaped asserts (`count_positives(vec![5,-2,3])`,
    // `reverse(vec![1,2])`); without it every array-in/array-out repo fn declined at mining.
    // Also accept a slice literal `&[..]` (a common assert arg when the fn takes `&[i64]`).
    let arr = t.strip_prefix("vec!").map(str::trim).unwrap_or(t);
    let arr = arr.strip_prefix('&').map(str::trim).unwrap_or(arr);
    if arr.starts_with('[') && arr.ends_with(']') {
        let inner = arr[1..arr.len() - 1].trim();
        if inner.is_empty() {
            return Some(Value::int_array(&[]));
        }
        let ints: Option<Vec<i64>> =
            inner.split(',').map(|x| x.trim().parse::<i64>().ok()).collect();
        if let Some(ints) = ints {
            return Some(Value::int_array(&ints));
        }
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
    // GATE against the failing test's asserts. This stage runs FIRST in the ladder and grounds via
    // the low-confidence prose bridge; without this, a mis-grounding pre-empts the example-verified
    // stages with a patch that doesn't actually satisfy the failing test (the dominant repo-repair
    // failure mode). Decline when the synthesized program can't reproduce the mined I/O.
    if !synthesis_reproduces_failing_asserts(context, &repo_fn, &result.code) {
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
    // GATE against the failing test's asserts for the fn being added: the test that references
    // the missing fn IS its spec. A mis-grounded body would append a wrong fn that pre-empts the
    // example-verified stages. Decline when the synthesized fn can't reproduce the mined I/O.
    if !synthesis_reproduces_failing_asserts(context, &fn_name, &result.code) {
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

/// LEVER D — extract-helper refactor (conservative, sound, string-level).
///
/// Parses `extract ... into [a] [helper|function] [called|named] <NAME>`, finds a NON-TRIVIAL
/// exact-duplicated PARENTHESISED integer-arithmetic expression `(E)` occurring >=2 times in ONE
/// repo fn body, and hoists it into `fn <NAME>(<free i64 locals>) -> i64 { E }`, replacing every
/// `(E)` occurrence with a call. Behaviour-preservation is gated downstream by the real cargo test.
///
/// SOUND-BY-CONSTRUCTION scope (declines otherwise — never emits wrong code):
///   - Only PURE i64 arithmetic expressions: every char in `E` is an identifier, an integer
///     literal, whitespace, or one of `+ - * / % ( )`. No calls / indexing / method calls /
///     comparisons — so `E` is provably `i64` and the helper's return type is KNOWN.
///   - Every free identifier in `E` must be a KNOWN `i64` param or `let NAME: i64` local of the
///     fn. Any unknown identifier (a const, a non-`i64` local, a call target) → DECLINE (the free-
///     variable / type analysis is unclear, so we refuse rather than guess).
///   - The expression must be genuinely duplicated (>=2 identical `(E)` groups) and non-trivial
///     (contains an arithmetic operator and >=1 free variable).
///
/// Full syn-AST extraction (arbitrary types, non-parenthesised spans, statement hoisting) is the
/// documented FOLLOW-UP; this covers the common "hoist a repeated arithmetic sub-expression" case.
pub fn try_extract_helper_patch(context: &RepairContext, description: &str) -> Option<RepairPatch> {
    let name = parse_extract_helper_name(description)?;
    // Collision guard: never shadow an existing fn (would be an E0428 or a silent capture).
    let defined: Vec<String> = context
        .files
        .iter()
        .filter(|f| f.path.ends_with(".rs"))
        .flat_map(|f| defined_fn_names(f.text.as_deref().unwrap_or("")))
        .collect();
    if defined.iter().any(|d| *d == name) {
        return None;
    }

    for file in &context.files {
        if !file.path.ends_with(".rs") {
            continue;
        }
        let text = file.text.as_deref().unwrap_or("");
        // Operate only on the code BEFORE the first test module — the oracle is never edited.
        let split = text.find("#[cfg(test)]").unwrap_or(text.len());
        let (prefix, suffix) = text.split_at(split);
        for (fn_name, fn_text) in split_top_level_functions(prefix) {
            let Some(body) = fn_body(&fn_text) else {
                continue;
            };
            let known = known_i64_names(&fn_text, &fn_name, &body);
            let Some((paren_expr, free_vars)) = find_duplicated_i64_expr(&body, &known) else {
                continue;
            };
            let params =
                free_vars.iter().map(|v| format!("{v}: i64")).collect::<Vec<_>>().join(", ");
            let inner = paren_expr[1..paren_expr.len() - 1].trim();
            let helper = format!("fn {name}({params}) -> i64 {{ {inner} }}");
            let call = format!("{name}({})", free_vars.join(", "));
            let new_body = body.replace(&paren_expr, &call);
            if new_body == body {
                continue;
            }
            let new_prefix = replace_body_only(prefix, &fn_name, &new_body).ok()?;
            let with_helper = insert_before_fn_def(&new_prefix, &fn_name, &helper);
            let new_text = format!("{with_helper}{suffix}");
            if new_text == text {
                continue;
            }
            return Some(
                RepairPatch::new()
                    .with_edit(RepairEdit::new(
                        file.path.clone(),
                        text.to_string(),
                        new_text,
                        format!("extract-helper refactor: hoist duplicated `{paren_expr}` into fn {name}"),
                    ))
                    .with_metadata("proposer", "nl_extract_helper")
                    .with_metadata("helper_name", name.clone()),
            );
        }
    }
    None
}

/// Parse the helper NAME from an "extract ... into [a] [helper|function] [called|named] <NAME>"
/// instruction. Requires both an `extract` token AND a `helper`/`function` token so plain prose
/// never triggers it. NAME follows `called`/`named` when present, else the token after
/// `helper`/`function`. Returns `None` if no clean identifier name is found.
fn parse_extract_helper_name(description: &str) -> Option<String> {
    let toks: Vec<&str> = description
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| !t.is_empty())
        .collect();
    let lower_toks: Vec<String> = toks.iter().map(|t| t.to_lowercase()).collect();
    if !lower_toks.iter().any(|t| t == "extract") {
        return None;
    }
    if !lower_toks.iter().any(|t| t == "helper" || t == "function") {
        return None;
    }
    let name = if let Some(p) = lower_toks.iter().rposition(|t| t == "called" || t == "named") {
        toks.get(p + 1).copied()
    } else {
        let p = lower_toks.iter().position(|t| t == "helper" || t == "function")?;
        toks.get(p + 1).copied()
    }?;
    if !is_plain_ident(name) {
        return None;
    }
    const FILLER: &[&str] =
        &["a", "an", "the", "it", "into", "helper", "function", "called", "named", "this", "that"];
    if FILLER.contains(&name.to_lowercase().as_str()) {
        return None;
    }
    Some(name.to_string())
}

/// Names in the fn known to be exactly `i64`: params typed `i64` plus `let [mut] NAME: i64 = ...`
/// locals. Used to decide which identifiers in a candidate expression are free `i64` variables.
fn known_i64_names(
    fn_text: &str,
    fn_name: &str,
    body: &str,
) -> std::collections::HashSet<String> {
    let mut set = std::collections::HashSet::new();
    if let Some(params) = fn_header_params(fn_text, fn_name) {
        let idents = parse_param_idents(&params);
        let types = parse_param_types(&params);
        for (n, t) in idents.iter().zip(types.iter()) {
            if t.trim() == "i64" {
                set.insert(n.clone());
            }
        }
    }
    for line in body.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("let ") {
            let rest = rest.trim();
            let rest = rest.strip_prefix("mut ").unwrap_or(rest);
            if let Some((name, ty_and)) = rest.split_once(':') {
                let name = name.trim();
                let ty = ty_and.split('=').next().unwrap_or("").trim();
                if ty == "i64" && is_plain_ident(name) {
                    set.insert(name.to_string());
                }
            }
        }
    }
    set
}

/// True for a plain Rust identifier (`[A-Za-z_][A-Za-z0-9_]*`).
fn is_plain_ident(s: &str) -> bool {
    let mut cs = s.chars();
    matches!(cs.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Every balanced `(...)` group (full text, parens included) in `s`, including nested groups.
fn balanced_paren_groups(s: &str) -> Vec<String> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    for i in 0..b.len() {
        if b[i] == b'(' {
            let mut depth = 0i32;
            for j in i..b.len() {
                match b[j] {
                    b'(' => depth += 1,
                    b')' => {
                        depth -= 1;
                        if depth == 0 {
                            out.push(s[i..=j].to_string());
                            break;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    out
}

/// If `paren_expr` (a full `(E)` group) is a PURE `i64` arithmetic expression whose every free
/// identifier is in `known`, return its free variables in first-appearance order. Otherwise `None`
/// — declining calls, indexing, comparisons, casts, unknown identifiers, or trivial (operator-free)
/// expressions, so the extracted helper's `-> i64` return type is always provably correct.
fn qualify_i64_expr(
    paren_expr: &str,
    known: &std::collections::HashSet<String>,
) -> Option<Vec<String>> {
    if paren_expr.len() < 2 {
        return None;
    }
    let inner = &paren_expr[1..paren_expr.len() - 1];
    // Char whitelist: identifiers, digits, whitespace, and pure-arithmetic operators only.
    if !inner
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c.is_whitespace() || "+-*/%()".contains(c))
    {
        return None;
    }
    // Must contain at least one arithmetic operator (non-trivial: not a bare `(x)`).
    if !inner.chars().any(|c| "+-*/%".contains(c)) {
        return None;
    }
    let bytes = inner.as_bytes();
    let mut free: Vec<String> = Vec::new();
    let mut i = 0usize;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_')
            {
                i += 1;
            }
            let ident = &inner[start..i];
            // A call target (`ident(`) is unsupported — return type unknown → decline.
            if inner[i..].trim_start().starts_with('(') {
                return None;
            }
            if known.contains(ident) {
                if !free.iter().any(|f| f == ident) {
                    free.push(ident.to_string());
                }
            } else {
                // Unknown identifier (const / non-i64 local / keyword) → decline.
                return None;
            }
        } else {
            i += 1;
        }
    }
    if free.is_empty() {
        return None;
    }
    Some(free)
}

/// The best duplicated, qualifying `(E)` group in `body`: the one appearing the most times
/// (>=2), tie-broken by longest text. Returns `(paren_expr, free_vars)` or `None`.
fn find_duplicated_i64_expr(
    body: &str,
    known: &std::collections::HashSet<String>,
) -> Option<(String, Vec<String>)> {
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for g in balanced_paren_groups(body) {
        *counts.entry(g).or_insert(0) += 1;
    }
    let mut best: Option<(String, usize, Vec<String>)> = None;
    for (text, count) in counts {
        if count < 2 {
            continue;
        }
        let Some(free) = qualify_i64_expr(&text, known) else {
            continue;
        };
        let better = match &best {
            None => true,
            Some((bt, bc, _)) => count > *bc || (count == *bc && text.len() > bt.len()),
        };
        if better {
            best = Some((text, count, free));
        }
    }
    best.map(|(t, _, f)| (t, f))
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
pub(crate) fn reshape_to_repo_signature(old_text: &str, repo_fn: &str, synthesized: &str) -> Option<String> {
    let r = reshape_to_repo_signature_inner(old_text, repo_fn, synthesized);
    if std::env::var_os("NSYNTH_DEBUG_RESHAPE").is_some() {
        eprintln!("=== RESHAPE repo_fn={repo_fn}\n--- SYNTHESIZED ---\n{synthesized}\n--- RESULT ---\n{}\n===",
            r.as_deref().unwrap_or("<None>"));
    }
    r
}

fn reshape_to_repo_signature_inner(old_text: &str, repo_fn: &str, synthesized: &str) -> Option<String> {
    // Normalize the generated Rust FIRST (the same pass the fixture scaffolder runs):
    // Mog-lowered bodies carry `: Vec<i64> = []` (E0308), bare `.len`, uncast `arr[i]`,
    // moved-Vec loops. Without this a Vec-RETURNING synth (reverse) reached cargo with
    // `let mut out: Vec<i64> = [];` -> type mismatch. Scalar folds happened to avoid the
    // `= []` shape, which is why only the collection cases regressed.
    let synthesized = &crate::agent::repo::gencode_normalize::normalize_component(synthesized);
    let fns = split_top_level_functions(synthesized);

    let repo_has_fn = old_text.contains(&format!("fn {repo_fn}"));

    // ARITY GUARD: decline when the synthesized ENTRY fn takes a different number of
    // params than the repo fn. A body written for the wrong arity can never satisfy the
    // signature — e.g. "maximum of two numbers" (repo `max_of(a,b)`) resolves via the NL
    // intent to a 1-arg ARRAY max (`array_max(xs)`); reshaping it onto 2 scalar params
    // yields a type mismatch that cargo rejects, and returning it here short-circuits the
    // proposer ladder before the test-mined stage (which mines the 2-arg asserts and
    // solves it correctly). Declining lets the ladder fall through to that stage.
    if repo_has_fn {
        let repo_n = fn_header_params(old_text, repo_fn).map(|p| parse_param_idents(&p).len());
        let entry = fns
            .iter()
            .find(|(n, _)| n == repo_fn)
            .or_else(|| fns.iter().find(|(_, t)| t.trim_start().starts_with("pub ")))
            .or_else(|| fns.last());
        let entry_n = entry.and_then(|(n, t)| fn_header_params(t, n).map(|p| parse_param_idents(&p).len()));
        if let (Some(a), Some(b)) = (repo_n, entry_n) {
            if a != b {
                return None;
            }
        }
    }

    // ARG-ORDER SWAP: when the synth entry's param TYPES form the same multiset as the repo
    // target's but in a DIFFERENT, unambiguous order, a positional rename would MIS-BIND params
    // (e.g. repo `(k: i64, xs: Vec<i64>)` vs a library op `(arr, k)`). Emit a wrapper that calls
    // the synth impl with the repo params reordered BY TYPE instead. `arg_reorder_permutation`
    // returns `None` (fall through to the normal path) for identity order, ambiguous same-typed
    // params, or arity/type mismatches — so this never fires on the aligned case.
    if repo_has_fn {
        if let Some(entry_idx) = fns
            .iter()
            .position(|(n, _)| n == repo_fn)
            .or_else(|| fns.iter().position(|(_, t)| t.trim_start().starts_with("pub ")))
            .or_else(|| (!fns.is_empty()).then(|| fns.len() - 1))
        {
            let synth_types = fn_header_params(&fns[entry_idx].1, &fns[entry_idx].0)
                .map(|p| parse_param_types(&p))
                .unwrap_or_default();
            let repo_types = fn_header_params(old_text, repo_fn)
                .map(|p| parse_param_types(&p))
                .unwrap_or_default();
            if let Some(perm) = arg_reorder_permutation(&synth_types, &repo_types) {
                if let Some(new_text) =
                    emit_arg_reorder_wrapper(old_text, repo_fn, &fns, entry_idx, &perm)
                {
                    return Some(new_text);
                }
            }
        }
    }

    // Slice adapters bridge the synthesizer's owned `Vec<i64>` params to a repo `&[i64]` slice
    // signature (see slice_param_adapters). Only meaningful when the repo fn exists.
    let adapters = if repo_has_fn { slice_param_adapters(old_text, repo_fn) } else { String::new() };
    // i64->bool adapter: the solver expresses a boolean predicate as an i64-returning function
    // (0/1, or truthy-nonzero — e.g. is_positive found as `reverse_number`, whose value is nonzero
    // iff n>0). The repo signature declares `-> bool`, so the i64 body type-mismatches. When the
    // repo fn returns bool but the synthesized entry returns i64, wrap each bare `return EXPR;` as
    // `return (EXPR) != 0;` (Mog's truthiness). Same-type (bool->bool combinators) are untouched.
    let needs_bool_adapt = fn_return_type(old_text, repo_fn).as_deref() == Some("bool")
        && first_fn_name_in_source(synthesized)
            .and_then(|n| fn_return_type(synthesized, &n))
            .as_deref()
            == Some("i64");

    if fns.len() > 1 {
        // Pick the ENTRY function: the repo fn by name, else the `pub` entry the
        // synthesizer emits (helpers are non-pub), else the last. Falling straight to
        // `fns.len()-1` mis-selected a private HELPER as main when the synthesized entry
        // was named for the op (e.g. `pub fn array_max` calling `fn max_except_last`,
        // repo fn `biggest`) — the helper got renamed to the repo fn while the real
        // entry was kept verbatim, producing a duplicate/dangling def (E0428).
        let main_idx = fns
            .iter()
            .position(|(name, _)| name == repo_fn)
            .or_else(|| {
                fns.iter()
                    .position(|(_, text)| text.trim_start().starts_with("pub "))
            })
            .unwrap_or(fns.len() - 1);
        let mut helpers = String::new();
        let mut emitted: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for (k, (name, text)) in fns.iter().enumerate() {
            if k == main_idx {
                continue;
            }
            // De-dup helpers so a composition that lists the same helper twice (e.g. `array_sum`
            // used at two call sites) does not emit two `fn array_sum` definitions (E0428), and
            // skip any helper the repo file ALREADY defines (same E0428) — a synthesized helper
            // that clashes with an existing one is redundant, not a second definition.
            if !emitted.insert(name.as_str()) {
                continue;
            }
            if repo_has_fn && old_text.contains(&format!("fn {name}(")) {
                continue;
            }
            helpers.push_str(text.trim());
            helpers.push_str("\n\n");
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
    if needs_bool_adapt {
        body = wrap_bare_returns_as_bool(&body);
    }
    // A repo param the (renamed) body MUTATES in place — sort's `a[i] = ..`, an in-place reverse —
    // needs a `mut` binding, but the repo signature declares the param immutable. Shadow it as
    // mutable at the top of the body (mirrors the slice `.to_vec()` adapter). Owned params only;
    // `&[..]` slice params already get a fresh owned binding through `adapters` above.
    let repo_types = fn_header_params(old_text, repo_fn)
        .map(|p| parse_param_types(&p))
        .unwrap_or_default();
    let mut mut_shadows = String::new();
    for (name, ty) in repo_params.iter().zip(repo_types.iter()) {
        // `&[..]` and `&str` params already get a fresh owned binding via `adapters`; a plain
        // owned param that the body mutates needs a `mut` shadow.
        if !ty.contains("&[")
            && !ty.contains("&str")
            && crate::mog_transpile::param_is_mutated(&body, name)
        {
            mut_shadows.push_str(&format!("let mut {name} = {name};\n"));
        }
    }
    // Prepend slice adapters + mut shadows so the repo signature compiles against the Vec-based body.
    let body = format!("{adapters}{mut_shadows}{body}");
    replace_body_only(old_text, repo_fn, &body).ok()
}

/// VERBATIM repair: if the model returned a single full `pub fn NAME(...) -> RET { .. }`
/// whose signature (params + return type) matches the repo fn EXACTLY, replace the whole
/// repo fn definition with the model's text as-is — no body extraction, no param renaming,
/// no `is_plain_rust_body` gate — so a byte-for-byte-correct fn is never lost to the reshape
/// heuristics (which throw away the model text, extract only the body, positionally rename
/// idents, and can decline entirely). cargo still gates the whole patch afterwards, so a bad
/// verbatim swap fails exactly as reshape output would; strict signature matching keeps this
/// firing only when the model fn truly slots in, else we fall through to reshape.
fn verbatim_repo_fn_replacement(old_text: &str, repo_fn: &str, model_src: &str) -> Option<String> {
    let fns = split_top_level_functions(model_src);
    if fns.len() != 1 {
        return None; // single-fn output only; multi-fn goes through the reshape branch
    }
    let (name, text) = &fns[0];
    if name != repo_fn {
        return None; // must target the repo fn by name
    }
    if !old_text.contains(&format!("fn {repo_fn}")) {
        return None; // the repo must actually define it
    }
    // Exact signature match: whitespace-normalized params AND identical return type.
    let norm = |s: Option<String>| s.map(|p| p.split_whitespace().collect::<String>());
    let mp = norm(fn_header_params(model_src, repo_fn));
    let rp = norm(fn_header_params(old_text, repo_fn));
    if mp.is_none() || mp != rp {
        return None;
    }
    if fn_return_type(model_src, repo_fn) != fn_return_type(old_text, repo_fn) {
        return None;
    }
    // Apply the model's fn wholesale (ensure `pub` so an exported repo fn stays exported).
    let verbatim = ensure_pub_fn(text.trim());
    replace_function_body(old_text, repo_fn, &verbatim).ok()
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
/// MODEL SYNTHESIS (self-synthesis of novel logic): when the symbolic engine cannot synthesize a
/// mined problem, ask the served local model for a Mog program. INERT unless NSYNTH_LOCAL_LLM_URL is
/// set (`propose_program` returns `None`), so there is zero behaviour change by default. Best-of-N
/// with a rising-temperature schedule and concrete failure feedback (mirrors the flywheel's
/// model_teach); a proposal is accepted ONLY if — renamed to the repo fn — it reproduces EVERY mined
/// example AND clears the robustness floor (`verify_problem_code_strict`). The caller distils the
/// accepted program into the learned store, so the engine solves the same shape model-free next
/// time. The model never bypasses a gate: the guarantee is the example-reproduction + robustness
/// checks here plus the repair loop's cargo-test oracle, not the model.
fn try_model_synthesis(
    problem: &crate::benchmark::Problem,
    exs: &[crate::benchmark::Example],
    repo_fn: &str,
) -> Option<String> {
    let mut request = format!("{repo_fn}\n\nExamples:\n");
    for ex in exs {
        request.push_str(&format!("  {:?} -> {:?}\n", ex.inputs, ex.expected));
    }
    request.push_str("\nWrite the Mog function.");
    let mut prior: Option<(String, String)> = None;
    for &t in &[0.1f64, 0.3, 0.5, 0.7] {
        let p = prior.as_ref().map(|(c, e)| (c.as_str(), e.as_str()));
        // `?` short-circuits the whole helper when there is no endpoint (first call is None), so the
        // default (model-off) machine never spins the best-of loop.
        let code = crate::local_llm::propose_program(&request, p, t)?;
        let renamed = rename_first_fn(&code, repo_fn);
        if !crate::runtime::code_reproduces_examples(&renamed, exs) {
            prior = Some((code, "wrong output on one of the examples — fix the logic".to_string()));
            continue;
        }
        if crate::runtime::verify_problem_code_strict(problem, &renamed).is_err() {
            prior = Some((
                code,
                "crashes / misbehaves on nearby inputs — handle edge cases".to_string(),
            ));
            continue;
        }
        return Some(renamed);
    }
    None
}

/// The return type spelled in `fn NAME(..) -> RET {`. `None` if the fn or arrow is absent.
fn fn_return_type(code: &str, name: &str) -> Option<String> {
    let start = code.find(&format!("fn {name}("))?;
    let after = &code[start..];
    let arrow = after.find("->")?;
    let brace = after[arrow..].find('{')?;
    Some(after[arrow + 2..arrow + brace].trim().to_string())
}

/// Wrap each bare `return EXPR;` as `return (EXPR) != 0;` (Mog truthiness), so an i64 predicate
/// body satisfies a `-> bool` signature. `return true/false;` and bare `return;` are left as-is.
fn wrap_bare_returns_as_bool(body: &str) -> String {
    body.lines()
        .map(|line| {
            let t = line.trim_start();
            if let Some(rest) = t.strip_prefix("return ") {
                if let Some(expr) = rest.strip_suffix(';') {
                    let e = expr.trim();
                    if !e.is_empty() && e != "true" && e != "false" {
                        let indent = &line[..line.len() - t.len()];
                        return format!("{indent}return ({e}) != 0;");
                    }
                }
            }
            line.to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

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

/// ARG-ORDER reconciliation: when the synthesized entry fn's parameter TYPES form the same
/// multiset as the repo target's but in a DIFFERENT order, compute the reordering of the REPO
/// params that supplies the synth's positional arguments. `perm[i]` is the index into the repo
/// params feeding the synth's i-th parameter.
///
/// Returns `Some(perm)` ONLY when the mapping is a genuine, UNAMBIGUOUS, non-identity
/// permutation: every canonical type KIND is DISTINCT (so each synth slot maps to exactly one
/// repo param), the multisets match, and the order truly differs. Returns `None` — a clean
/// decline that keeps the caller's existing behaviour — when: arities differ, any type is
/// un-mappable, any KIND repeats (ambiguous: two params share a type, so a swap could silently
/// mis-bind them), or the orders already match (identity: the normal positional rename is
/// correct). Never guesses across an ambiguous mapping.
fn arg_reorder_permutation(synth_types: &[String], repo_types: &[String]) -> Option<Vec<usize>> {
    if synth_types.is_empty() || synth_types.len() != repo_types.len() {
        return None;
    }
    let synth_kinds: Vec<&'static str> = synth_types
        .iter()
        .map(|t| crate::library_probe::canonical_kind(t))
        .collect::<Option<_>>()?;
    let repo_kinds: Vec<&'static str> = repo_types
        .iter()
        .map(|t| crate::library_probe::canonical_kind(t))
        .collect::<Option<_>>()?;
    // Ambiguity guard: every KIND must be unique on BOTH sides. A repeated kind means a
    // reorder could silently swap two same-typed params without a compile error — decline.
    for kinds in [&synth_kinds, &repo_kinds] {
        let mut seen = std::collections::HashSet::new();
        for k in kinds.iter() {
            if !seen.insert(*k) {
                return None;
            }
        }
    }
    // Build the permutation: synth slot i wants the repo param whose kind matches. Distinct
    // kinds guarantee `position` is unique; a missing kind (multiset mismatch) declines.
    let mut perm = Vec::with_capacity(synth_kinds.len());
    for sk in &synth_kinds {
        perm.push(repo_kinds.iter().position(|rk| rk == sk)?);
    }
    // Identity (orders already align) → the normal rename path is correct; not our case.
    if perm.iter().enumerate().all(|(i, &p)| i == p) {
        return None;
    }
    Some(perm)
}

/// Emit an arg-order-reconciling WRAPPER: keep the synthesized entry fn as a private sibling
/// (renamed to `reordered_{repo_fn}` — a PREFIX so `find("fn {repo_fn}")` never lands on it), keep any other
/// synthesized helper fns as siblings, and rewrite the repo fn's body to call the sibling with
/// its own parameters reordered per `perm` (adding `.to_vec()` where the repo passes a slice into
/// the Vec-based impl). The repo signature is preserved verbatim, so the caller's tests keep
/// their call convention. Returns `None` if any structural step fails.
fn emit_arg_reorder_wrapper(
    old_text: &str,
    repo_fn: &str,
    fns: &[(String, String)],
    entry_idx: usize,
    perm: &[usize],
) -> Option<String> {
    let (entry_name, entry_text) = &fns[entry_idx];
    // Prefix (not suffix) the helper name: `replace_body_only` / `insert_before_fn_def` locate
    // a fn by the substring `fn {repo_fn}`, so a suffix name (`k_largest_reorder_impl`) is matched
    // FIRST when we later target `k_largest`, corrupting the impl instead of the repo fn. A
    // prefix (`reordered_k_largest`) never contains `fn k_largest`, so targeting stays exact.
    let impl_name = format!("reordered_{repo_fn}");
    // Rename ALL word-boundary occurrences of the entry name (declaration + any self-calls) so a
    // recursive op keeps working under the new name.
    let entry_renamed = rename_ident(entry_text, entry_name, &impl_name);

    // Sibling block: the renamed entry first, then every other synthesized helper verbatim.
    let mut helpers = String::new();
    helpers.push_str(entry_renamed.trim());
    helpers.push_str("\n\n");
    for (k, (_, text)) in fns.iter().enumerate() {
        if k != entry_idx {
            helpers.push_str(text.trim());
            helpers.push_str("\n\n");
        }
    }

    // Reordered call args from the repo's own parameters.
    let repo_params = fn_header_params(old_text, repo_fn)?;
    let repo_idents = parse_param_idents(&repo_params);
    let repo_types = parse_param_types(&repo_params);
    if repo_idents.len() != repo_types.len() || perm.len() != repo_idents.len() {
        return None;
    }
    let mut call_args = Vec::with_capacity(perm.len());
    for &ri in perm {
        let name = repo_idents.get(ri)?;
        let ty = repo_types.get(ri)?;
        // Bridge a slice repo param into the owned-Vec impl param.
        if ty.contains("&[") {
            call_args.push(format!("{name}.to_vec()"));
        } else {
            call_args.push(name.clone());
        }
    }
    let wrapper_body = format!("return {impl_name}({});", call_args.join(", "));

    let with_helpers = insert_before_fn_def(old_text, repo_fn, helpers.trim());
    replace_body_only(&with_helpers, repo_fn, &wrapper_body).ok()
}

/// `.to_vec()` shadow lines for each repo parameter whose type is a slice (`&[..]`), so the
/// Vec-based synthesized body compiles against a slice signature. A collection repo function
/// (`pub fn sum_of_evens(xs: &[i64]) -> i64`) keeps its slice signature (matching the test's
/// call convention) while the synthesized logic sees an owned `Vec` — the minimal bridge that
/// unblocks list-processing repo repairs without rewriting iteration.
fn slice_param_adapters(old_text: &str, repo_fn: &str) -> String {
    // Bridge a repo `&[i64]` slice PARAM to the synthesizer's owned `Vec<i64>` body by
    // shadowing the param: `let xs = xs.to_vec();`. This is safe for ANY return type —
    // it only rebinds the parameter, so a Vec-RETURNING fn (reverse -> Vec<i64>) still
    // returns its Vec unchanged. (Earlier this early-returned empty for Vec/[]-returning
    // fns to "avoid mangling the return", but the param shadow never touches the return;
    // skipping it left reverse's slice param bound to a Vec-based body -> E0308 type
    // mismatch. Only reverse has a &[i64] param among the fixtures, so this is surgical.)
    let params = fn_header_params(old_text, repo_fn).unwrap_or_default();
    let idents = parse_param_idents(&params);
    let types = parse_param_types(&params);
    let mut out = String::new();
    for (name, ty) in idents.iter().zip(types.iter()) {
        if ty.contains("&[") {
            out.push_str(&format!("let {name} = {name}.to_vec();\n"));
        } else if ty.contains("&str") {
            // Bridge a repo `&str` param to the synthesizer's owned `String` body (Mog `string`
            // lowers to `String`): shadow it as an owned `String`. Same shape as the slice
            // adapter — needed for string-transform ops (snake_to_camel) whose repo fn takes
            // `&str` while the op body operates on an owned string -> E0308 without this.
            out.push_str(&format!("let {name} = {name}.to_string();\n"));
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

    #[test]
    fn mutation_underflow_guard_clamps_a_usize_local() {
        // A usize local used later as `VAR - 1` (a real panic bug, e.g. bytesize's unit exponent) gets
        // the clamp mutation `let mut VAR = ...; if VAR == 0 { VAR = 1; }`. Emitted so it's tried before
        // the thousands of operator swaps a large file produces.
        let code = "impl X {\n  fn fmt(&self) -> String {\n    let exp = ((self.n as f64).ln() / 2.0) as usize;\n    format!(\"{}\", UNITS[exp - 1])\n  }\n}\n";
        let muts = generate_mutations(code);
        assert!(
            muts.iter().any(|m| m.contains("let mut exp =") && m.contains("if exp == 0 { exp = 1; }")),
            "no underflow-guard clamp among {} mutations",
            muts.len()
        );
    }

    #[test]
    fn function_span_at_line_returns_enclosing_fn_body() {
        // line 1: `fn a`, line 2: body, line 3: `}`, line 4: blank, line 5: `fn b`, line 6: body...
        let code = "fn a() -> i32 {\n    1\n}\n\nfn b() -> i32 {\n    2\n}\n";
        // A line inside `a` maps to the `fn a { .. }` span.
        let (s, e) = function_span_at_line(code, 2).expect("line 2 is in fn a");
        assert!(code[s..e].starts_with("fn a"), "got: {:?}", &code[s..e]);
        assert!(code[s..e].trim_end().ends_with('}'));
        assert!(!code[s..e].contains("fn b"));
        // A line inside `b` maps to `fn b`, not `fn a`.
        let (s2, e2) = function_span_at_line(code, 6).expect("line 6 is in fn b");
        assert!(code[s2..e2].starts_with("fn b"), "got: {:?}", &code[s2..e2]);
        // A blank line between functions encloses nothing.
        assert_eq!(function_span_at_line(code, 4), None);
        // Out-of-range line -> None (no panic).
        assert_eq!(function_span_at_line(code, 999), None);
    }

    #[test]
    fn localized_mutations_come_before_whole_file_and_stay_valid() {
        // Two functions; the bug (an operator to flip) is in the SECOND. Localizing to `sub` must put
        // its mutations first, and each spliced candidate must keep the first function intact.
        let code = "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n\nfn sub(a: i32, b: i32) -> i32 {\n    a + b\n}\n";
        let span = function_span_at_line(code, 6).expect("line 6 in fn sub");
        let localized = localized_then_full_mutations(code, Some(span));
        // The `a + b` -> `a - b` fix (in sub) appears, and among the FIRST candidates it keeps `add` whole.
        let head = &localized[..localized.len().min(8)];
        assert!(
            head.iter().any(|m| m.contains("fn add(a: i32, b: i32) -> i32 {\n    a + b")
                && m.contains("fn sub(a: i32, b: i32) -> i32 {\n    a - b")),
            "localized head did not fix sub while preserving add: {head:#?}"
        );
        // With no span it degrades to plain whole-file mutations (caller path unchanged)...
        let full_only = localized_then_full_mutations(code, None);
        assert_eq!(full_only.len(), generate_mutations(code).len());
        // ...and the span version is the function-variants PREPENDED to that whole-file set, so it is
        // strictly longer (the localized candidates are tried first, before the cap is spent).
        assert!(
            localized.len() > full_only.len(),
            "span version must prepend function-variants: {} vs {}",
            localized.len(),
            full_only.len()
        );
    }

    #[test]
    fn integer_literal_mutation_spares_const_declarations() {
        // A `const`/`static` declaration's value must NOT be mutated (corrupting a named constant to
        // pass a weak test is a wrong fix)...
        let decl = "pub const KIB: u64 = 1024;\nstatic MAX: i64 = 60;\n";
        let dm = generate_mutations(decl);
        assert!(
            !dm.iter().any(|m| m.contains("= 1023") || m.contains("= 1025") || m.contains("= 61") || m.contains("= 59")),
            "mutated a const/static declaration value"
        );
        // ...but a literal in an EXPRESSION (the legitimate wrong-constant / off-by-one class) still is.
        let expr = "pub fn kib(n: i64) -> i64 { n * 1000 }\n";
        let em = generate_mutations(expr);
        assert!(
            em.iter().any(|m| m.contains("n * 1001") || m.contains("n * 999")),
            "stopped mutating expression literals"
        );
    }

    #[test]
    fn bool_return_negation_toggles_the_predicate() {
        // Tail-expression form: `v.is_empty()` -> `!(v.is_empty())`.
        let a = "pub fn accepts(v: &[i64]) -> bool { v.is_empty() }\n";
        assert!(
            generate_mutations(a).iter().any(|m| m.contains("!(v.is_empty())")),
            "no negation of the tail predicate"
        );
        // `return EXPR;` form, and removing an existing `!`.
        let b = "pub fn ok(n: i64) -> bool { return !flag(n); }\n";
        assert!(
            generate_mutations(b).iter().any(|m| m.contains("return flag(n);")),
            "no removal of the stray negation"
        );
        // Must NOT fire on a non-bool return (no spurious `!` on an i64).
        let c = "pub fn add(a: i64, b: i64) -> i64 { a + b }\n";
        assert!(
            !generate_mutations(c).iter().any(|m| m.contains("!(")),
            "negated a non-bool return"
        );
    }

    #[test]
    fn assert_failure_localizes_to_the_named_production_fn_not_the_test_or_a_sibling() {
        // `subject` is the buggy fn; `reference` is a sibling the assert also calls. An assert failure
        // must localize to `subject` (named first in the assert), NOT the test and NOT `reference`.
        let code = "pub fn subject(a: i64) -> i64 { a + 1 }\n\
                    pub fn reference(a: i64) -> i64 { a - 1 }\n\n\
                    #[cfg(test)]\nmod tests {\n use super::*;\n #[test]\n fn t() {\n  \
                    assert_eq!(subject(5), reference(5));\n }\n}\n";
        // the assert line (1-indexed)
        let assert_line = code[..code.find("assert_eq!").unwrap()].matches('\n').count() + 1;
        let span = production_target_span(code, assert_line).expect("named-fn span");
        // the span starts at `fn subject` (the `pub ` prefix stays outside, preserved by the splice)
        assert!(code[span.0..span.1].starts_with("fn subject"), "got: {:?}", &code[span.0..span.1]);
        assert!(code[span.0..span.1].contains("a + 1") && !code[span.0..span.1].contains("reference"));
        // called_idents skips the macro and finds the free calls in order.
        assert_eq!(called_idents("assert_eq!(subject(5), reference(5));"), vec!["subject", "reference"]);
        assert_eq!(called_idents("let x = obj.method(3) + free(4);"), vec!["free"]);
    }

    #[test]
    fn localized_mutations_never_touch_the_test_module() {
        // An assert failure reports the ASSERT's line, which sits in `#[cfg(test)]`. A span there must
        // NOT localize (else it would mutate the test to "pass") — it degrades to whole-file, which
        // excludes tests. So the result equals the plain whole-file mutation set, and NO candidate
        // alters the test asserts.
        let code = "pub fn fits(x: i64, cap: i64) -> bool { x < cap }\n\n\
                    #[cfg(test)]\nmod tests {\n use super::*;\n #[test]\n fn t() {\n  \
                    assert_eq!(fits(5, 5), true);\n }\n}\n";
        // A line inside the test module.
        let test_line = code[..code.find("assert_eq!").unwrap()].matches('\n').count() + 1;
        let span = function_span_at_line(code, test_line);
        assert!(span.is_some(), "the assert line should resolve to the test fn span");
        let muts = localized_then_full_mutations(code, span);
        assert_eq!(
            muts.len(),
            generate_mutations(code).len(),
            "a test-region span must NOT prepend localized (test-mutating) candidates"
        );
        // Belt and suspenders: no candidate changes the asserted literal (would corrupt the test).
        assert!(
            muts.iter().all(|m| m.contains("assert_eq!(fits(5, 5), true)")),
            "a mutation altered the test assertion — never-wrong violation"
        );
    }

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

    #[test]
    fn asserted_call_name_recovers_the_dominant_called_function() {
        let mut counts = std::collections::HashMap::new();
        // call side is the NON-literal argument; the expected value may be on either side
        accumulate_asserted_call_names(
            "assert_eq!(count_positives(vec![5, -2, 3]), 2);\n\
             assert_eq!(0, count_positives(vec![-1, -2]));\n\
             assert_eq!(count_positives(vec![1]), 1);",
            &mut counts,
        );
        assert_eq!(counts.get("count_positives"), Some(&3), "dominant call recovered from both arg orders");
        // a bare literal on both sides records nothing; `vec![..]` is not a call
        let mut empty = std::collections::HashMap::new();
        accumulate_asserted_call_names("assert_eq!(1 + 1, 2); assert_eq!(vec![1], vec![1]);", &mut empty);
        assert!(empty.is_empty(), "no function call → no name; vec! macro is not a call: {empty:?}");
        // leading_call_ident: a call vs a non-call
        assert_eq!(leading_call_ident("count_positives(vec![1,2])").as_deref(), Some("count_positives"));
        assert_eq!(leading_call_ident("&foo(3)").as_deref(), Some("foo"));
        assert_eq!(leading_call_ident("some_var"), None);
        assert_eq!(leading_call_ident("42"), None);
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
            patch.metadata.iter().any(|(k, v)| k == "proposer" && v == "model_repair_multifn"),
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

    /// Repo repair reaching the OP LIBRARY through the never-wrong front door: a repo
    /// `count_positives` (filter+count) that the bare enumerator refuses to synthesize from its
    /// mined array asserts is solved because `verified_nl_router::answer` resolves the library op
    /// and verifies it against those same asserts. Exercises array-INPUT mining (parse_literal
    /// vec![..]) + array-output guard removal + the front-door proposer.
    #[test]
    fn test_mined_reaches_library_via_front_door_count_positives() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        std::env::remove_var("NSYNTH_LOCAL_LLM_URL");
        let root = std::env::temp_dir().join(format!("nsynth_frontdoor_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"fd\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        )
        .expect("cargo.toml");
        fs::write(
            root.join("src/lib.rs"),
            "pub fn count_positives(xs: Vec<i64>) -> i64 {\n    0\n}\n\n#[cfg(test)]\nmod tests {\n    use super::count_positives;\n    #[test]\n    fn t() {\n        assert_eq!(count_positives(vec![5, -2, 3, -4, 5]), 3);\n        assert_eq!(count_positives(vec![-1, -2, -3]), 0);\n        assert_eq!(count_positives(vec![1, 2, 3, 4]), 4);\n    }\n}\n",
        )
        .expect("lib.rs");
        let task = RepoTaskSpec {
            id: "fd".into(),
            repo: root.to_string_lossy().to_string(),
            kind: RepoTaskKind::BugFix,
            issue: "nl: count how many elements are positive".into(),
            test_command: "cargo test".into(),
            allowed_files: vec!["src/**".into()],
            max_iterations: 2,
            hardness: HardnessProfile::for_expected_tier(HardnessTier::SingleFileBug),
            signals: Vec::new(),
        };
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("ctx");
        let patch = try_test_mined_synthesis_patch(
            &task,
            &context,
            "count how many elements are positive",
            None,
        )
        .expect("front-door proposer should solve count_positives (library op via mined asserts)");
        assert!(
            patch
                .metadata
                .iter()
                .any(|(k, v)| k == "synthesis_method" && v.starts_with("front-door:")),
            "should resolve through the front door, not raw synthesis: {:?}",
            patch.metadata
        );
        assert!(
            patch.edits.iter().any(|e| e.path == "src/lib.rs" && !e.new_text.contains("    0\n}")),
            "stub body must be replaced with a real count"
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// FEATURE-ADD (missing function): the failing asserts CALL `count_positives` but no file
    /// DEFINES it — the crate won't even compile. The old inline test-mined path always ran
    /// `reshape_to_repo_signature` as a body REPLACEMENT and had no branch for a function that
    /// isn't there. Delegating to `synthesize_mined_for_fn` takes its `repo_has_fn == false`
    /// branch, which synthesizes the verified op and APPENDS it to the file (feature-add) rather
    /// than replacing existing contents. The library op `count_positives` shares the repo fn's
    /// name, so the redundant strict re-verify (`problem.name`) still matches — this test isolates
    /// the append behavior, not the orthogonal name-mismatch case. Verifies the patch preserves
    /// the original asserts (append, not overwrite) and adds a real `fn count_positives`.
    #[test]
    fn test_mined_feature_adds_missing_function_by_appending() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        std::env::remove_var("NSYNTH_LOCAL_LLM_URL");
        let root = std::env::temp_dir().join(format!("nsynth_featadd_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"fa\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        )
        .expect("cargo.toml");
        // No definition of `count_positives` anywhere — only the failing asserts that call it.
        let original = "#[cfg(test)]\nmod tests {\n    use super::count_positives;\n    #[test]\n    fn t() {\n        assert_eq!(count_positives(vec![5, -2, 3, -4, 5]), 3);\n        assert_eq!(count_positives(vec![-1, -2, -3]), 0);\n        assert_eq!(count_positives(vec![1, 2, 3, 4]), 4);\n    }\n}\n";
        fs::write(root.join("src/lib.rs"), original).expect("lib.rs");
        let task = RepoTaskSpec {
            id: "fa".into(),
            repo: root.to_string_lossy().to_string(),
            kind: RepoTaskKind::BugFix,
            issue: "nl: count how many elements are positive".into(),
            test_command: "cargo test".into(),
            allowed_files: vec!["src/**".into()],
            max_iterations: 2,
            hardness: HardnessProfile::for_expected_tier(HardnessTier::SingleFileBug),
            signals: Vec::new(),
        };
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("ctx");
        let patch = try_test_mined_synthesis_patch(
            &task,
            &context,
            "count how many elements are positive",
            None,
        )
        .expect("feature-add: missing `count_positives` should be synthesized and appended");
        let edit = patch
            .edits
            .iter()
            .find(|e| e.path == "src/lib.rs")
            .expect("edit must target src/lib.rs");
        assert!(
            edit.new_text.contains("fn count_positives"),
            "a real `fn count_positives` must be added, got: {}",
            edit.new_text
        );
        assert!(
            edit.new_text.contains("assert_eq!(count_positives(vec![5, -2, 3, -4, 5]), 3)"),
            "append must PRESERVE the original asserts, not overwrite the file"
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// NAME-MISMATCH RESCUE: the repo fn is `double`, but "double the input value" resolves through
    /// the front door to the library op `times_two` — a DIFFERENT canonical name. The redundant
    /// strict re-verify runs a `main` that calls `problem.name` (== `double`), so before the fix the
    /// `fn times_two` body spuriously failed that verify and the stage declined, even though the
    /// router had already verified `times_two` reproduces the mined asserts. The single-fn entry is
    /// now renamed to `double` before the verify, so the stub is repaired.
    #[test]
    fn test_mined_repairs_when_library_op_name_differs_from_repo_fn() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        std::env::remove_var("NSYNTH_LOCAL_LLM_URL");
        let root = std::env::temp_dir().join(format!("nsynth_namemismatch_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"nm\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        )
        .expect("cargo.toml");
        // `double` is a stub returning its input; the failing asserts pin 2*n.
        fs::write(
            root.join("src/lib.rs"),
            "pub fn double(n: i64) -> i64 {\n    n\n}\n\n#[cfg(test)]\nmod tests {\n    use super::double;\n    #[test]\n    fn t() {\n        assert_eq!(double(2), 4);\n        assert_eq!(double(3), 6);\n        assert_eq!(double(4), 8);\n    }\n}\n",
        )
        .expect("lib.rs");
        let task = RepoTaskSpec {
            id: "nm".into(),
            repo: root.to_string_lossy().to_string(),
            kind: RepoTaskKind::BugFix,
            issue: "nl: double the input value".into(),
            test_command: "cargo test".into(),
            allowed_files: vec!["src/**".into()],
            max_iterations: 2,
            hardness: HardnessProfile::for_expected_tier(HardnessTier::SingleFileBug),
            signals: Vec::new(),
        };
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("ctx");
        let patch = try_test_mined_synthesis_patch(&task, &context, "double the input value", None)
            .expect("front-door op `times_two` should repair repo fn `double` after name canonicalization");
        let edit = patch
            .edits
            .iter()
            .find(|e| e.path == "src/lib.rs")
            .expect("edit targets src/lib.rs");
        assert!(
            edit.new_text.contains("fn double(n: i64) -> i64"),
            "repo signature (name `double`) preserved: {}",
            edit.new_text
        );
        assert!(
            edit.new_text.contains('2') && !edit.new_text.contains("times_two"),
            "body is the 2*n behavior renamed onto `double`, not a stray `times_two`: {}",
            edit.new_text
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// MULTI-FUNCTION targeting: a crate with TWO broken stubs (`inc`, `dbl`) in ONE file. The
    /// description resolves to a SINGLE fn, so without failure-awareness both iterations would
    /// re-target that one fn and the other stub would never be reached. The failing assert's
    /// `file:line` (from `FailureAnalysis`) names the fn cargo is currently failing on, so the stage
    /// targets THAT fn — the same call with the same description but a different failing line repairs
    /// a different function. This is what lets the edit→test→repair loop converge on a multi-fn crate.
    #[test]
    fn test_mined_targets_the_failing_function_in_a_multi_fn_crate() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        std::env::remove_var("NSYNTH_LOCAL_LLM_URL");
        let root = std::env::temp_dir().join(format!("nsynth_multifn_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"mf\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        )
        .expect("cargo.toml");
        // inc asserts are lines 14-16, dbl asserts are lines 17-19 (1-based).
        let src = "pub fn inc(n: i64) -> i64 {\n    n\n}\n\npub fn dbl(n: i64) -> i64 {\n    n\n}\n\n#[cfg(test)]\nmod tests {\n    use super::{inc, dbl};\n    #[test]\n    fn t() {\n        assert_eq!(inc(2), 3);\n        assert_eq!(inc(5), 6);\n        assert_eq!(inc(9), 10);\n        assert_eq!(dbl(2), 4);\n        assert_eq!(dbl(5), 10);\n        assert_eq!(dbl(9), 18);\n    }\n}\n";
        fs::write(root.join("src/lib.rs"), src).expect("lib.rs");
        let task = RepoTaskSpec {
            id: "mf".into(),
            repo: root.to_string_lossy().to_string(),
            kind: RepoTaskKind::BugFix,
            issue: "nl: increment a number".into(),
            test_command: "cargo test".into(),
            allowed_files: vec!["src/**".into()],
            max_iterations: 3,
            hardness: HardnessProfile::for_expected_tier(HardnessTier::SingleFileBug),
            signals: Vec::new(),
        };
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("ctx");
        let fail_at = |line: u32| FailureAnalysis {
            kind: FailureKind::TestFailure,
            file: Some("src/lib.rs".into()),
            line: Some(line),
            message: String::new(),
            likely_cause: String::new(),
            suggested_action: String::new(),
        };
        // SAME description, SAME context — only the failing line differs. Failure on a `dbl` assert
        // (line 17) must target `dbl`; failure on an `inc` assert (line 14) must target `inc`.
        let dbl_fail = fail_at(17);
        let body_dbl = try_test_mined_synthesis_patch(
            &task,
            &context,
            "increment a number",
            Some(&dbl_fail),
        )
        .and_then(|p| p.edits.into_iter().find(|e| e.path == "src/lib.rs").map(|e| e.new_text))
        .expect("failing `dbl` assert should target `dbl`");
        // dbl repaired to a doubling; inc LEFT AS A STUB — proves the failing assert (not the
        // description, which resolves to inc) chose the target.
        assert!(
            body_dbl.contains("fn dbl(n: i64) -> i64 {\n    return 2 * n;"),
            "the `dbl` stub is repaired to 2*n: {body_dbl}"
        );
        assert!(
            body_dbl.contains("pub fn inc(n: i64) -> i64 {\n    n\n}"),
            "`inc` must stay a stub when the failure targets `dbl`: {body_dbl}"
        );

        let inc_fail = fail_at(14);
        let body_inc = try_test_mined_synthesis_patch(
            &task,
            &context,
            "increment a number",
            Some(&inc_fail),
        )
        .and_then(|p| p.edits.into_iter().find(|e| e.path == "src/lib.rs").map(|e| e.new_text))
        .expect("failing `inc` assert should target `inc`");
        assert!(
            body_inc.contains("fn inc(") && body_inc.contains("+ 1;"),
            "the `inc` stub is repaired to n+1: {body_inc}"
        );
        assert!(
            body_inc.contains("pub fn dbl(n: i64) -> i64 {\n    n\n}"),
            "`dbl` must stay a stub when the failure targets `inc`: {body_inc}"
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// COMPOUND repair grounded in the REPO, not the prose. A crate with TWO broken functions
    /// (`add`, `sub`), each isolated by its own asserts inside one test. The failing test names both
    /// functions, so `try_multifn_mined_patch` fixes BOTH in a single atomic patch (independent of
    /// `max_iterations`). The "how many functions" decision comes from the asserted call-names — never
    /// a regex on the word "and" in the issue text.
    #[test]
    fn multifn_repairs_every_function_the_failing_asserts_name_in_one_patch() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        std::env::remove_var("NSYNTH_LOCAL_LLM_URL");
        let root = std::env::temp_dir().join(format!("nsynth_compound_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"cp\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        )
        .expect("cargo.toml");
        // add asserts on lines 14-15, sub asserts on 16-17 (1-based).
        let src = "pub fn add(a: i64, b: i64) -> i64 {\n    0\n}\n\npub fn sub(a: i64, b: i64) -> i64 {\n    0\n}\n\n#[cfg(test)]\nmod tests {\n    use super::{add, sub};\n    #[test]\n    fn t() {\n        assert_eq!(add(2, 3), 5);\n        assert_eq!(add(10, 1), 11);\n        assert_eq!(sub(9, 4), 5);\n        assert_eq!(sub(10, 3), 7);\n    }\n}\n";
        fs::write(root.join("src/lib.rs"), src).expect("lib.rs");
        let task = RepoTaskSpec {
            id: "cp".into(),
            repo: root.to_string_lossy().to_string(),
            kind: RepoTaskKind::BugFix,
            issue: "nl: fix the add and subtract functions".into(),
            test_command: "cargo test".into(),
            allowed_files: vec!["src/**".into()],
            max_iterations: 1, // ONE iteration must fix BOTH — proves the single atomic patch
            hardness: HardnessProfile::for_expected_tier(HardnessTier::SingleFileBug),
            signals: Vec::new(),
        };
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("ctx");
        let fail = FailureAnalysis {
            kind: FailureKind::TestFailure,
            file: Some("src/lib.rs".into()),
            line: Some(14), // the first failing assert (add)
            message: String::new(),
            likely_cause: String::new(),
            suggested_action: String::new(),
        };
        let patch = try_multifn_mined_patch(
            &task,
            &context,
            "fix the add and subtract functions",
            Some(&fail),
        )
        .expect("compound stage should repair both `add` and `sub` in one patch");
        assert_eq!(patch.edits.len(), 1, "one atomic single-file edit");
        assert!(
            patch.metadata.iter().any(|(k, v)| k == "functions_fixed" && v == "2"),
            "two functions repaired: {:?}",
            patch.metadata
        );
        let new_text = &patch.edits[0].new_text;
        // both stubs replaced: add -> a+b, sub -> a-b (neither left as `0`)
        assert!(
            new_text.contains("fn add(a: i64, b: i64) -> i64 {") && new_text.contains("a + b"),
            "`add` repaired to a+b: {new_text}"
        );
        assert!(
            new_text.contains("fn sub(a: i64, b: i64) -> i64 {") && new_text.contains("a - b"),
            "`sub` repaired to a-b: {new_text}"
        );
        assert!(
            !new_text.contains("-> i64 {\n    0\n}"),
            "no stub body (`0`) may remain — both functions fixed: {new_text}"
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// DISTINGUISHING NEGATIVE: a single-function crate must NOT trigger the compound stage — there is
    /// exactly one asserted function, so `try_multifn_mined_patch` declines (returns None) and the
    /// single-function path handles it. Proves the stage keys off the REPO's function count, so it can
    /// never over-split an ordinary one-function repair.
    #[test]
    fn multifn_declines_on_a_single_function_crate() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        std::env::remove_var("NSYNTH_LOCAL_LLM_URL");
        let root = std::env::temp_dir().join(format!("nsynth_compound_neg_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"cn\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        )
        .expect("cargo.toml");
        fs::write(
            root.join("src/lib.rs"),
            "pub fn mystery(n: i64) -> i64 {\n    n\n}\n\n#[cfg(test)]\nmod tests {\n    use super::mystery;\n    #[test]\n    fn t() {\n        assert_eq!(mystery(2), 5);\n        assert_eq!(mystery(3), 7);\n        assert_eq!(mystery(4), 9);\n    }\n}\n",
        )
        .expect("lib.rs");
        let task = RepoTaskSpec {
            id: "cn".into(),
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
        let fail = FailureAnalysis {
            kind: FailureKind::TestFailure,
            file: Some("src/lib.rs".into()),
            line: Some(8),
            message: String::new(),
            likely_cause: String::new(),
            suggested_action: String::new(),
        };
        assert!(
            try_multifn_mined_patch(&task, &context, "fix the mystery function", Some(&fail))
                .is_none(),
            "a single-function crate must NOT trigger the compound stage (no over-split)"
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// LEVER C END TO END: a repo `k_largest(xs: Vec<i64>, k: i64) -> Vec<i64>` stub is repaired by
    /// resolving the unique library op `k_largest` (a 2-arg `(array, scalar) -> array` param-carrying
    /// op) whose behaviour reproduces the mined asserts, then reshaping it onto the repo signature.
    /// This is the `(Vec, k) -> Vec` shape that previously declined at reshape/resolution.
    #[test]
    fn test_mined_k_largest_two_arg_array_scalar_to_array_end_to_end() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        std::env::remove_var("NSYNTH_LOCAL_LLM_URL");
        let root = std::env::temp_dir().join(format!("nsynth_klargest_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"kl\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        )
        .expect("cargo.toml");
        fs::write(
            root.join("src/lib.rs"),
            "pub fn k_largest(xs: Vec<i64>, k: i64) -> Vec<i64> {\n    Vec::new()\n}\n\n#[cfg(test)]\nmod tests {\n    use super::k_largest;\n    #[test]\n    fn t() {\n        assert_eq!(k_largest(vec![5, 1, 9, 3], 2), vec![9, 5]);\n        assert_eq!(k_largest(vec![4, 4, 2, 8, 1], 3), vec![8, 4, 4]);\n        assert_eq!(k_largest(vec![3, 1], 9), vec![3, 1]);\n    }\n}\n",
        )
        .expect("lib.rs");
        let task = RepoTaskSpec {
            id: "kl".into(),
            repo: root.to_string_lossy().to_string(),
            kind: RepoTaskKind::BugFix,
            issue: "nl: the k largest elements".into(),
            test_command: "cargo test".into(),
            allowed_files: vec!["src/**".into()],
            max_iterations: 2,
            hardness: HardnessProfile::for_expected_tier(HardnessTier::SingleFileBug),
            signals: Vec::new(),
        };
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("ctx");
        // Aligned (array, scalar) -> array shape resolves via the front door (prose names the op)
        // OR the behaviour probe (mined asserts pin the unique library op), then reshapes cleanly.
        let patch = try_test_mined_synthesis_patch(&task, &context, "the k largest elements", None)
            .or_else(|| {
                crate::library_probe::try_library_behavior_patch(
                    &task,
                    &context,
                    "the k largest elements",
                )
            })
            .expect("k_largest (Vec,k)->Vec must resolve end to end");
        let edit = patch
            .edits
            .iter()
            .find(|e| e.path == "src/lib.rs")
            .expect("edit to lib.rs");
        assert!(
            !edit.new_text.contains("Vec::new()\n}"),
            "stub body must be replaced by the real k-largest logic: {}",
            edit.new_text
        );
        // The reshaped body must keep the repo signature and compile-shape (sort/push present).
        assert!(
            edit.new_text.contains("fn k_largest(xs: Vec<i64>, k: i64) -> Vec<i64>"),
            "repo signature preserved: {}",
            edit.new_text
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// ARG-ORDER SWAP adapter, tested directly: a clean two-param swap of distinct types yields
    /// the reordering permutation; ambiguous (repeated type) and different-arity cases decline;
    /// identity order declines (the normal rename path is correct there).
    #[test]
    fn arg_reorder_permutation_accepts_clean_swap_declines_ambiguous() {
        let vec_i64 = "Vec<i64>".to_string();
        let i64_t = "i64".to_string();
        // synth (i64, Vec<i64>) vs repo (Vec<i64>, i64): synth slot 0 (int) -> repo idx 1,
        // synth slot 1 (array) -> repo idx 0  => perm [1, 0].
        assert_eq!(
            arg_reorder_permutation(&[i64_t.clone(), vec_i64.clone()], &[vec_i64.clone(), i64_t.clone()]),
            Some(vec![1, 0]),
            "distinct-type swap must produce the reordering permutation"
        );
        // Identity order: already aligned, decline (normal path handles it).
        assert_eq!(
            arg_reorder_permutation(&[vec_i64.clone(), i64_t.clone()], &[vec_i64.clone(), i64_t.clone()]),
            None,
            "identity order must decline"
        );
        // Ambiguous: two params share a type (both i64) -> cannot pin a mapping -> decline.
        assert_eq!(
            arg_reorder_permutation(&[i64_t.clone(), i64_t.clone()], &[i64_t.clone(), i64_t.clone()]),
            None,
            "repeated type must decline (ambiguous)"
        );
        // Different arity: decline outright.
        assert_eq!(
            arg_reorder_permutation(&[i64_t.clone()], &[vec_i64.clone(), i64_t.clone()]),
            None,
            "arity mismatch must decline"
        );
        // Un-mappable type: decline.
        assert_eq!(
            arg_reorder_permutation(&["Foo".to_string(), i64_t.clone()], &[i64_t.clone(), "Foo".to_string()]),
            None,
            "un-mappable type must decline"
        );
    }

    /// Reshape emits a WRAPPER when the synthesized entry's param order differs from the repo's.
    /// Repo `k_largest(k: i64, xs: Vec<i64>)` (scalar-first) + a library-shaped synth
    /// `k_largest(arr: Vec<i64>, k: i64)` (array-first) must NOT be positionally renamed (which
    /// would mis-bind `arr`/`k`); instead a `reordered_{repo_fn}` sibling is emitted and the repo fn
    /// calls it with the params reordered by type: `reordered_k_largest(xs, k)`.
    #[test]
    fn reshape_emits_arg_reorder_wrapper_for_scalar_first_repo() {
        let old = "pub fn k_largest(k: i64, xs: Vec<i64>) -> Vec<i64> {\n    Vec::new()\n}\n";
        let synth = "pub fn k_largest(arr: Vec<i64>, k: i64) -> Vec<i64> {\n    let mut tmp = arr;\n    tmp.sort();\n    let mut out: Vec<i64> = Vec::new();\n    let n = tmp.len() as i64;\n    let mut lim = k;\n    if lim > n { lim = n; }\n    let mut i = 0i64;\n    while i < lim { out.push(tmp[(n - 1 - i) as usize]); i = i + 1; }\n    return out;\n}\n";
        let new = reshape_to_repo_signature(old, "k_largest", synth)
            .expect("scalar-first repo must reshape via arg-order wrapper");
        // Repo signature preserved verbatim.
        assert!(
            new.contains("pub fn k_largest(k: i64, xs: Vec<i64>) -> Vec<i64>"),
            "repo signature must be preserved: {new}"
        );
        // A renamed sibling impl exists (PREFIX name — never collides with the repo fn target).
        assert!(new.contains("fn reordered_k_largest("), "impl sibling must be emitted: {new}");
        // The impl KEEPS the real body (the earlier bug replaced it with a self-call, losing this).
        assert!(new.contains(".sort()"), "impl must retain the real body, not a self-call: {new}");
        // The REPO fn body is the reordered call into the impl (array `xs` first, `k` second).
        assert!(
            new.contains("return reordered_k_largest(xs, k)"),
            "repo fn must call the impl with args reordered by type (xs, k): {new}"
        );
        // The repo fn is no longer the empty stub.
        assert!(!new.contains("-> Vec<i64> {\n    Vec::new()\n}"), "repo stub must be replaced: {new}");
    }

    /// Reshape adapter for a SLICE-typed scalar-first repo param: the array param is `&[i64]`,
    /// so the reordered call must bridge it into the Vec-based impl via `.to_vec()`.
    #[test]
    fn reshape_arg_reorder_wrapper_bridges_slice_param() {
        let old = "pub fn take_top(k: i64, xs: &[i64]) -> Vec<i64> {\n    Vec::new()\n}\n";
        let synth = "pub fn take_top(arr: Vec<i64>, k: i64) -> Vec<i64> {\n    let mut tmp = arr;\n    tmp.truncate(k as usize);\n    return tmp;\n}\n";
        let new = reshape_to_repo_signature(old, "take_top", synth)
            .expect("slice scalar-first repo must reshape via arg-order wrapper");
        assert!(
            new.contains("return reordered_take_top(xs.to_vec(), k)"),
            "slice repo param must be bridged with .to_vec() in the reordered call: {new}"
        );
        assert!(new.contains(".truncate("), "impl must retain the real body: {new}");
    }

    /// A synthesized composition that emits the SAME helper twice (e.g. `array_sum` at two call
    /// sites) must reshape to a SINGLE `fn array_sum` — two definitions are E0428 and the compile
    /// gate rejects the whole patch (observed: prefix-sums). De-dup by helper name.
    #[test]
    fn reshape_dedups_duplicate_synthesized_helpers() {
        let old = "pub fn prefix_sums(xs: Vec<i64>) -> Vec<i64> {\n    Vec::new()\n}\n";
        let helper = "fn array_sum(a: Vec<i64>) -> i64 {\n    let mut s: i64 = 0;\n    for e in a.iter().copied() {\n        s = s + e;\n    }\n    return s;\n}\n";
        let synth = format!(
            "pub fn prefix_sums(arr: Vec<i64>) -> Vec<i64> {{\n    let mut out: Vec<i64> = Vec::new();\n    return out;\n}}\n\n{helper}\n{helper}"
        );
        let new = reshape_to_repo_signature(old, "prefix_sums", &synth)
            .expect("multi-fn composition must reshape");
        assert_eq!(
            new.matches("fn array_sum(").count(),
            1,
            "the array_sum helper must appear exactly once (E0428 otherwise): {new}"
        );
    }

    /// A synthesized helper that already exists in the repo file is skipped (E0428 otherwise).
    #[test]
    fn reshape_skips_helper_already_in_repo() {
        let old = "pub fn array_sum(a: Vec<i64>) -> i64 {\n    0\n}\n\npub fn prefix_sums(xs: Vec<i64>) -> Vec<i64> {\n    Vec::new()\n}\n";
        let synth = "pub fn prefix_sums(arr: Vec<i64>) -> Vec<i64> {\n    let mut out: Vec<i64> = Vec::new();\n    return out;\n}\n\nfn array_sum(a: Vec<i64>) -> i64 {\n    let mut s: i64 = 0;\n    for e in a.iter().copied() {\n        s = s + e;\n    }\n    return s;\n}\n";
        let new = reshape_to_repo_signature(old, "prefix_sums", synth).expect("reshape");
        assert_eq!(
            new.matches("fn array_sum(").count(),
            1,
            "the pre-existing array_sum must not be re-emitted: {new}"
        );
    }

    /// A repo string-transform fn that takes `&str` (idiomatic) must bridge to the op's owned
    /// `String` body via a `let s = s.to_string();` shadow — E0308 otherwise (observed: snake->camel
    /// repo fns). The repo signature (`&str`) is preserved; the body operates on the owned copy.
    #[test]
    fn reshape_bridges_str_param_to_owned_string() {
        let old = "pub fn snake_to_camel(s: &str) -> String {\n    String::new()\n}\n";
        let synth = "pub fn snake_to_camel(s: String) -> String {\n    let mut out = String::new();\n    for ch in s.chars() {\n        out.push(ch);\n    }\n    return out;\n}\n";
        let new = reshape_to_repo_signature(old, "snake_to_camel", synth).expect("reshape");
        assert!(
            new.contains("let s = s.to_string();"),
            "&str param must be bridged to an owned String: {new}"
        );
        assert!(
            new.contains("pub fn snake_to_camel(s: &str) -> String"),
            "repo &str signature must be preserved: {new}"
        );
    }

    // ---- Lever D: extract-helper refactor ----

    #[test]
    fn extract_helper_name_parses_and_declines() {
        assert_eq!(
            parse_extract_helper_name("extract the duplicated expression into a helper called top"),
            Some("top".to_string())
        );
        assert_eq!(
            parse_extract_helper_name("extract it into a function named area"),
            Some("area".to_string())
        );
        // No helper/function token → not an extract-helper instruction.
        assert_eq!(parse_extract_helper_name("extract the value into top"), None);
        // No extract token at all.
        assert_eq!(parse_extract_helper_name("make a helper called top"), None);
        // Name would be a filler word → decline.
        assert_eq!(parse_extract_helper_name("extract into a helper called it"), None);
    }

    #[test]
    fn qualify_i64_expr_accepts_pure_arith_declines_calls_and_unknowns() {
        let mut known = std::collections::HashSet::new();
        known.insert("a".to_string());
        known.insert("b".to_string());
        known.insert("c".to_string());
        // Pure i64 arithmetic, free vars in first-appearance order (deduped).
        assert_eq!(
            qualify_i64_expr("(a * b + c)", &known),
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()])
        );
        assert_eq!(qualify_i64_expr("(a * a)", &known), Some(vec!["a".to_string()]));
        // A call target → unknown return type → decline.
        assert_eq!(qualify_i64_expr("(f(a) + b)", &known), None);
        // Unknown identifier (not a known i64 local) → decline.
        assert_eq!(qualify_i64_expr("(a + z)", &known), None);
        // Comparison / non-arithmetic char → decline (would not be i64).
        assert_eq!(qualify_i64_expr("(a < b)", &known), None);
        // Trivial (no operator) → decline.
        assert_eq!(qualify_i64_expr("(a)", &known), None);
    }

    /// END TO END structural: a repo fn with a duplicated pure-i64 expression + an "extract it into
    /// a helper called <NAME>" Refactor instruction yields a patch that ADDS `fn <NAME>` and
    /// replaces >=2 occurrences with a call — the repo signature and test module untouched.
    #[test]
    fn extract_helper_hoists_duplicated_expression() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        let root = std::env::temp_dir().join(format!("nsynth_extract_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"ex\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        )
        .expect("cargo.toml");
        // `(a * b + c)` is duplicated across two lets; both must become `top(a, b, c)`.
        fs::write(
            root.join("src/lib.rs"),
            "pub fn score(a: i64, b: i64, c: i64) -> i64 {\n    let x = (a * b + c);\n    let y = (a * b + c);\n    x + y\n}\n\n#[cfg(test)]\nmod tests {\n    use super::score;\n    #[test]\n    fn t() {\n        assert_eq!(score(2, 3, 1), 14);\n    }\n}\n",
        )
        .expect("lib.rs");
        let task = RepoTaskSpec {
            id: "ex".into(),
            repo: root.to_string_lossy().to_string(),
            kind: RepoTaskKind::Refactor,
            issue: "nl: extract the duplicated expression into a helper called top".into(),
            test_command: "cargo test".into(),
            allowed_files: vec!["src/**".into()],
            max_iterations: 2,
            hardness: HardnessProfile::for_expected_tier(HardnessTier::SingleFileBug),
            signals: Vec::new(),
        };
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("ctx");
        let patch = try_extract_helper_patch(
            &context,
            "extract the duplicated expression into a helper called top",
        )
        .expect("extract-helper must hoist the duplicated expression");
        assert!(
            patch.metadata.iter().any(|(k, v)| k == "proposer" && v == "nl_extract_helper"),
            "extract-helper proposer must own the patch: {:?}",
            patch.metadata
        );
        let edit = patch.edits.iter().find(|e| e.path == "src/lib.rs").expect("edit lib.rs");
        assert!(edit.new_text.contains("fn top(a: i64, b: i64, c: i64) -> i64"), "{}", edit.new_text);
        assert!(
            edit.new_text.matches("top(a, b, c)").count() >= 2,
            ">=2 call sites must replace the duplicated expression: {}",
            edit.new_text
        );
        assert!(
            !edit.new_text.contains("(a * b + c)"),
            "the duplicated literal expression must be gone from the body: {}",
            edit.new_text
        );
        // The test module (oracle) is untouched.
        assert!(edit.new_text.contains("assert_eq!(score(2, 3, 1), 14);"), "{}", edit.new_text);
        // Ignore the task binding (kind-guard is exercised by the supervisor); silence unused.
        let _ = &task;
        let _ = fs::remove_dir_all(&root);
    }

    /// Declines when the duplicated expression references an UNKNOWN (non-i64) local — the free-
    /// variable / type analysis is unclear, so no patch is emitted rather than wrong code.
    #[test]
    fn extract_helper_declines_when_analysis_unclear() {
        let root = std::env::temp_dir().join(format!("nsynth_extract_dec_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"exd\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        )
        .expect("cargo.toml");
        // `s` is a String local (untyped-annotation, non-i64): the duplicated `(s.len() + a)`
        // contains a method call and an unknown local → must decline.
        fs::write(
            root.join("src/lib.rs"),
            "pub fn f(a: i64) -> usize {\n    let s = String::new();\n    let x = (s.len() + a as usize);\n    let y = (s.len() + a as usize);\n    x + y\n}\n",
        )
        .expect("lib.rs");
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("ctx");
        assert!(
            try_extract_helper_patch(&context, "extract it into a helper called g").is_none(),
            "must decline on unclear (non-i64 / method-call) free-variable analysis"
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// Array-OUTPUT repo fn via the front door: "reverse a list" (`Vec<i64> -> Vec<i64>`) is a
    /// library op reached through mined `vec![..]` asserts, now that the array-output guard is
    /// removed and reshape handles the Vec-return lowering. If reshape can't fit this signature
    /// the proposer declines gracefully (returns None) rather than emitting a bad patch.
    #[test]
    fn test_mined_array_output_reverse_via_front_door() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        std::env::remove_var("NSYNTH_LOCAL_LLM_URL");
        let root = std::env::temp_dir().join(format!("nsynth_fdrev_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"fdr\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        )
        .expect("cargo.toml");
        fs::write(
            root.join("src/lib.rs"),
            "pub fn reverse_list(xs: Vec<i64>) -> Vec<i64> {\n    xs\n}\n\n#[cfg(test)]\nmod tests {\n    use super::reverse_list;\n    #[test]\n    fn t() {\n        assert_eq!(reverse_list(vec![1, 2, 3]), vec![3, 2, 1]);\n        assert_eq!(reverse_list(vec![5, 6]), vec![6, 5]);\n        assert_eq!(reverse_list(vec![9]), vec![9]);\n    }\n}\n",
        )
        .expect("lib.rs");
        let task = RepoTaskSpec {
            id: "fdr".into(),
            repo: root.to_string_lossy().to_string(),
            kind: RepoTaskKind::BugFix,
            issue: "nl: reverse a list".into(),
            test_command: "cargo test".into(),
            allowed_files: vec!["src/**".into()],
            max_iterations: 2,
            hardness: HardnessProfile::for_expected_tier(HardnessTier::SingleFileBug),
            signals: Vec::new(),
        };
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("ctx");
        // Documents the CURRENT capability: a produced patch must come via the front door and
        // replace the identity stub; a None (reshape can't fit) is an honest decline, not a bug.
        if let Some(patch) =
            try_test_mined_synthesis_patch(&task, &context, "reverse a list", None)
        {
            assert!(
                patch
                    .metadata
                    .iter()
                    .any(|(k, v)| k == "synthesis_method" && v.starts_with("front-door:")),
                "array-output solve must be front-door-sourced: {:?}",
                patch.metadata
            );
            eprintln!("REVERSE_ARRAY_OUTPUT: front-door patch produced (array-output reshape works)");
        } else {
            eprintln!("REVERSE_ARRAY_OUTPUT: declined (array-output reshape still a gap; safe)");
        }
        let _ = fs::remove_dir_all(&root);
    }

    /// LADDER PRE-EMPTION GUARD: `try_real_synthesis_patch` grounds via the low-confidence prose
    /// bridge on the intent's OWN examples. When that grounding does not reproduce the failing
    /// test's asserts (a scalar `double`-grounding against an array-fold task), it must now
    /// DECLINE — so it can no longer return a wrong patch that pre-empts the example-verified
    /// stages. Without the gate it returned a mis-grounded patch here.
    #[test]
    fn real_synthesis_declines_when_grounding_misses_the_failing_asserts() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        std::env::remove_var("NSYNTH_LOCAL_LLM_URL");
        let root = std::env::temp_dir().join(format!("nsynth_laddergate_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"lg\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        )
        .expect("cargo.toml");
        // Asserts encode an ARRAY FOLD (sum); the prose names a scalar `double` — a mis-grounding.
        fs::write(
            root.join("src/lib.rs"),
            "pub fn total(xs: Vec<i64>) -> i64 {\n    0\n}\n\n#[cfg(test)]\nmod tests {\n    use super::total;\n    #[test]\n    fn t() {\n        assert_eq!(total(vec![1, 2, 3]), 6);\n        assert_eq!(total(vec![10, 20]), 30);\n        assert_eq!(total(vec![5]), 5);\n    }\n}\n",
        )
        .expect("lib.rs");
        let task = RepoTaskSpec {
            id: "lg".into(),
            repo: root.to_string_lossy().to_string(),
            kind: RepoTaskKind::BugFix,
            issue: "nl: double a number".into(),
            test_command: "cargo test".into(),
            allowed_files: vec!["src/**".into()],
            max_iterations: 2,
            hardness: HardnessProfile::for_expected_tier(HardnessTier::SingleFileBug),
            signals: Vec::new(),
        };
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("ctx");
        // The mis-grounded scalar-double program cannot reproduce total(vec![..]) == sum, so the
        // gate declines (returns None) instead of pre-empting the proven stages.
        assert!(
            try_real_synthesis_patch(&task, &context, "double a number").is_none(),
            "real-synthesis must decline a grounding that fails the failing test's asserts"
        );
        let _ = fs::remove_dir_all(&root);
    }

    /// The shared prose-grounding gate (used by all three synthesis stages): a candidate that
    /// reproduces the failing test's mined asserts passes; a mis-grounding is rejected; a fn with
    /// no asserts is permissive. Deterministic — no bridge/solver in the loop.
    #[test]
    fn synthesis_gate_accepts_reproducing_rejects_mismatched() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        let root = std::env::temp_dir().join(format!("nsynth_gate_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("src")).expect("mkdir");
        fs::write(
            root.join("Cargo.toml"),
            "[package]\nname = \"g\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        )
        .expect("cargo.toml");
        fs::write(
            root.join("src/lib.rs"),
            "pub fn total(xs: Vec<i64>) -> i64 {\n    0\n}\n\n#[cfg(test)]\nmod tests {\n    use super::total;\n    #[test]\n    fn t() {\n        assert_eq!(total(vec![1, 2, 3]), 6);\n        assert_eq!(total(vec![10, 20]), 30);\n        assert_eq!(total(vec![5]), 5);\n    }\n}\n",
        )
        .expect("lib.rs");
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("ctx");
        let sum_mog = "fn total(arr: [i64]) -> i64 {\n    s: i64 = 0;\n    for e in arr {\n        s = s + e;\n    }\n    return s;\n}\n";
        let max_mog = "fn total(arr: [i64]) -> i64 {\n    m: i64 = arr[0];\n    for e in arr {\n        if e > m {\n            m = e;\n        }\n    }\n    return m;\n}\n";
        assert!(
            synthesis_reproduces_failing_asserts(&context, "total", sum_mog),
            "a sum reproduces the sum asserts -> gate passes"
        );
        assert!(
            !synthesis_reproduces_failing_asserts(&context, "total", max_mog),
            "a max does NOT reproduce the sum asserts -> gate rejects the mis-grounding"
        );
        assert!(
            synthesis_reproduces_failing_asserts(&context, "no_such_fn", max_mog),
            "no asserts for the fn -> permissive (prior behavior preserved)"
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

    // ── gated MODEL-INTENT stage: inert-by-default + verify-gated (no server) ──

    /// Build an IN-MEMORY repair context (no disk) around a single source file, so
    /// the model-intent core can be exercised without a live model or a real crate.
    fn mi_context(text: &str) -> RepairContext {
        RepairContext {
            root: "/nonexistent/nsynth-mi-root".into(),
            files: vec![crate::agent::repo::RepairFile {
                path: "src/lib.rs".into(),
                bytes: text.len(),
                lines: text.lines().count(),
                text: Some(text.to_string()),
            }],
        }
    }

    fn mi_task(id: &str) -> RepoTaskSpec {
        RepoTaskSpec {
            id: id.into(),
            repo: "/nonexistent/nsynth-mi-root".into(),
            kind: RepoTaskKind::BugFix,
            issue: "nl: double the number".into(),
            test_command: "cargo test".into(),
            allowed_files: vec!["src/**".into()],
            max_iterations: 2,
            hardness: HardnessProfile::for_expected_tier(HardnessTier::SingleFileBug),
            signals: Vec::new(),
        }
    }

    /// GATE (i) — INERT BY DEFAULT: with `NSYNTH_LOCAL_LLM_URL` unset the model-intent
    /// lane returns `None` immediately (never reaching the network layer) and stages
    /// nothing. This is the zero-default-change guarantee.
    #[test]
    fn model_intent_lane_is_inert_without_url() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        std::env::remove_var("NSYNTH_LOCAL_LLM_URL");
        let ctx = mi_context("pub fn twice(n: i64) -> i64 {\n    n\n}\n");
        let task = mi_task("mi-inert");
        discard_model_distillation(&task.id);
        assert!(
            try_model_intent_patch(&task, &ctx, "double the number").is_none(),
            "gated model-intent lane must be inert without NSYNTH_LOCAL_LLM_URL"
        );
        assert!(
            take_model_distillation(&task.id).is_none(),
            "an inert lane must stage no distillation candidate"
        );
    }

    /// GATE (ii) — VERIFY-GATED: a deliberately-WRONG model spec (a stub for the model's
    /// output — no server needed) is REJECTED by the engine's held-out oracle and never
    /// promoted to a patch (and nothing is staged for distillation). A CORRECT spec, by
    /// contrast, is synthesized + verified into a repo-signature patch and stages a
    /// distillation candidate whose program reproduces the examples.
    #[test]
    fn model_intent_core_rejects_wrong_spec_and_verifies_correct_spec() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        use crate::benchmark::{Example, Value};
        let ctx = mi_context("pub fn twice(n: i64) -> i64 {\n    n\n}\n");

        // WRONG: identity on the seed, but a POISONED held-out row the seed-synthesized
        // program cannot reproduce — the never-wrong held-out gate rejects it.
        let wrong = vec![
            Example { inputs: vec![Value::Int(0)], expected: Value::Int(0) },
            Example { inputs: vec![Value::Int(1)], expected: Value::Int(1) },
            Example { inputs: vec![Value::Int(2)], expected: Value::Int(2) },
            Example { inputs: vec![Value::Int(3)], expected: Value::Int(3) },
            Example { inputs: vec![Value::Int(10)], expected: Value::Int(999) },
        ];
        let wtask = mi_task("mi-wrong");
        discard_model_distillation(&wtask.id);
        assert!(
            try_model_intent_patch_from_spec(&wtask, &ctx, "identity", wrong).is_none(),
            "a held-out-inconsistent model spec must be REJECTED, never promoted"
        );
        assert!(
            take_model_distillation(&wtask.id).is_none(),
            "a rejected spec must stage NO distillation candidate"
        );

        // CORRECT: doubling. The engine synthesizes 2n, reproduces the held-out row, and
        // passes strict-verify → a verified patch onto the repo `twice` signature.
        let good = vec![
            Example { inputs: vec![Value::Int(1)], expected: Value::Int(2) },
            Example { inputs: vec![Value::Int(2)], expected: Value::Int(4) },
            Example { inputs: vec![Value::Int(3)], expected: Value::Int(6) },
            Example { inputs: vec![Value::Int(4)], expected: Value::Int(8) },
            Example { inputs: vec![Value::Int(6)], expected: Value::Int(12) },
        ];
        let gtask = mi_task("mi-correct");
        discard_model_distillation(&gtask.id);
        let patch = try_model_intent_patch_from_spec(&gtask, &ctx, "double the number", good)
            .expect("a correct model spec yields a verified patch");
        assert!(
            patch.metadata.iter().any(|(k, v)| k == "proposer" && v == "model_intent"),
            "patch is tagged as the model-intent lane: {:?}",
            patch.metadata
        );
        assert!(
            patch.edits.iter().any(|e| e.new_text.contains("fn twice(n: i64)")),
            "the repo fn signature is preserved by the reshape: {:?}",
            patch.edits.iter().map(|e| e.new_text.clone()).collect::<Vec<_>>()
        );
        // An accepted-shaped solve stages a distillation candidate whose ENGINE program
        // reproduces the examples — exactly what `record_proposed_op` would absorb.
        let (problem, code) = take_model_distillation(&gtask.id)
            .expect("a verified model-intent solve stages a distill candidate");
        assert!(
            crate::runtime::code_reproduces_examples(&code, &problem.examples),
            "the staged program reproduces its examples: {code}"
        );
    }

    /// The distillation plumbing is single-shot and INERT when nothing was staged (the
    /// default-run case): `distill_accepted_model_solve` is a no-op with no candidate,
    /// staging round-trips the exact `(problem, code)`, and `discard` clears without
    /// recording. No env vars touched → no cross-test races.
    #[test]
    fn model_distillation_plumbing_single_shot_and_inert_when_empty() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        use crate::benchmark::{Example, Problem, Value};
        let id = format!("mi-plumbing-{}", std::process::id());
        discard_model_distillation(&id);
        assert!(
            !distill_accepted_model_solve(&id),
            "no staged candidate -> distilling an accepted solve is a no-op"
        );
        let sig: &'static str = Box::leak("fn f(n: i64) -> i64".to_string().into_boxed_str());
        let problem = Problem {
            name: "f".into(),
            signature: sig,
            examples: vec![Example { inputs: vec![Value::Int(2)], expected: Value::Int(7) }],
            ..Default::default()
        };
        let code = "fn f(n: i64) -> i64 {\n    return n * 3 + 1;\n}\n".to_string();
        stage_model_distillation(&id, problem.clone(), code.clone());
        let taken = take_model_distillation(&id).expect("staged candidate present");
        assert_eq!(taken.1, code, "code round-trips through the stage");
        assert_eq!(taken.0.examples, problem.examples, "problem round-trips through the stage");
        assert!(
            take_model_distillation(&id).is_none(),
            "single-shot: the candidate is consumed"
        );
        stage_model_distillation(&id, problem, code);
        discard_model_distillation(&id);
        assert!(
            take_model_distillation(&id).is_none(),
            "discard clears the stage without recording"
        );
    }


    #[test]
    fn mutation_generator_covers_operator_swaps_assignment_and_offbyone_and_spares_the_test() {
        let code = "pub fn f(n: i64) -> i64 {\n    let mut s = 0;\n    let mut i = 1;\n    while i < n { s = i; }\n    s\n}\n#[cfg(test)]\nmod tests { fn t() { assert_eq!(f(5), 15); } }\n";
        let muts = generate_mutations(code);
        // operator swap `<` -> `<=`
        assert!(muts.iter().any(|m| m.contains("while i <= n")), "no <= mutation");
        // assignment `=` -> `+=` (the struct-method / accumulator fix)
        assert!(muts.iter().any(|m| m.contains("s += i;")), "no += mutation");
        // off-by-one on a constant
        assert!(muts.iter().any(|m| m.contains("let mut i = 2;") || m.contains("let mut i = 0;")), "no const +-1");
        // the test module is NEVER mutated (would be cheating).
        assert!(
            muts.iter().all(|m| m.contains("assert_eq!(f(5), 15)")),
            "a mutation altered the test assertion"
        );
    }

    #[test]
    fn stub_fills_generate_stateful_method_bodies_from_struct_fields() {
        // A `&mut self` setter stub -> `self.v = x` (among candidates).
        let setter = "pub struct S { pub v: i64 }\nimpl S { pub fn set(&mut self, x: i64) {} }\n";
        let f1 = generate_stub_fills(setter);
        assert!(f1.iter().any(|m| m.contains("self.v = x;")), "no setter body: {f1:?}");
        // A Vec field -> push.
        let pusher = "pub struct L { pub items: Vec<i64> }\nimpl L { pub fn add(&mut self, x: i64) {} }\n";
        assert!(generate_stub_fills(pusher).iter().any(|m| m.contains("self.items.push(x)")), "no push body");
        // A getter stub -> return a field.
        let getter = "pub struct C { pub n: i64 }\nimpl C { pub fn get(&self) -> i64 {} }\n";
        assert!(generate_stub_fills(getter).iter().any(|m| m.contains("return self.n;")), "no getter body");
        // A no-param mutator (inc) -> `self.n += 1`.
        let inc = "pub struct C { pub n: i64 }\nimpl C { pub fn inc(&mut self) {} }\n";
        assert!(generate_stub_fills(inc).iter().any(|m| m.contains("self.n += 1;")), "no inc body");
    }

    #[test]
    fn mutation_swaps_compound_assignment() {
        // `self.total += x` (wrong-op struct bug injected as `-=`) must be recoverable both ways.
        assert!(generate_mutations("fn f(){ self.total -= x; }\n").iter().any(|m| m.contains("self.total += x")), "no -= -> +=");
        assert!(generate_mutations("fn f(){ self.total += x; }\n").iter().any(|m| m.contains("self.total -= x")), "no += -> -=");
    }

    #[test]
    fn mutation_operators_are_space_insensitive() {
        // UNSPACED assignment (the regression): `self.n=x` must yield `self.n+=x` and `self.n-=x`.
        let unspaced = "impl C { fn add(&mut self, x: i64) { self.n=x; } }\n";
        let m = generate_mutations(unspaced);
        assert!(m.iter().any(|s| s.contains("self.n+=x")), "no += for unspaced assign: had {}", m.len());
        assert!(m.iter().any(|s| s.contains("self.n-=x")), "no -= for unspaced assign");
        // UNSPACED arithmetic `w+h` -> `w-h`, `w*h`.
        let arith = generate_mutations("fn a(w: i64, h: i64) -> i64 { w+h }\n");
        assert!(arith.iter().any(|s| s.contains("w-h")), "no - for unspaced add");
        assert!(arith.iter().any(|s| s.contains("w*h")), "no * for unspaced add");
        // A generic `Vec<i64>` must NOT be mangled into `Vec<=i64>` / `Vec>i64>`.
        let generic = generate_mutations("fn f(v: Vec<i64>) -> i64 { 0 }\n");
        assert!(!generic.iter().any(|s| s.contains("Vec<=") || s.contains("Vec>")), "generic got mutated");
        // Arrow `->` and match `=>` must survive (no `-` / `=` operator match on them).
        let arrows = generate_mutations("fn f(x: i64) -> i64 { match x { 0 => 1, _ => x } }\n");
        assert!(arrows.iter().all(|s| s.contains("-> i64")), "return arrow was mutated");
    }

    #[test]
    fn mutation_method_name_swaps() {
        let code = "pub fn f(s: String) -> String { s.to_lowercase() }\n";
        assert!(generate_mutations(code).iter().any(|m| m.contains(".to_uppercase()")), "no case swap");
    }

    #[test]
    fn bounded_product_caps_and_covers() {
        let lists = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["x".to_string(), "y".to_string()],
        ];
        let full = bounded_product(&lists, 99);
        assert_eq!(full.len(), 4, "2x2 product");
        assert_eq!(full[0], vec!["a".to_string(), "x".to_string()], "first combo is all-first-candidates");
        assert!(bounded_product(&lists, 2).len() <= 2, "cap respected");
        assert_eq!(bounded_product(&[], 9), vec![Vec::<String>::new()]);
    }

    #[test]
    fn mh_fingerprint_stable_for_same_choices() {
        let make = || FileHoles {
            path: "src/lib.rs".into(),
            abs: PathBuf::from("/tmp/x"),
            orig: String::new(),
            tail: String::new(),
            code: String::new(),
            holes: Vec::new(),
            choice: vec!["self.a += x;".into(), "self.b".into()],
        };
        let a = mh_files_fingerprint(&[make()]);
        let b = mh_files_fingerprint(&[make()]);
        assert_eq!(a, b);
        let mut other = make();
        other.choice[0] = "self.a = x;".into();
        assert_ne!(a, mh_files_fingerprint(&[other]));
    }

    #[test]
    fn scan_holes_is_struct_aware_across_multiple_structs() {
        // A method's candidates must use ITS impl struct's fields, not a flat union of all structs.
        let code = "pub struct Item { pub title: String, pub priority: i64 }\npub struct TodoList { pub items: Vec<Item> }\nimpl TodoList {\n pub fn add(&mut self, title: String, priority: i64) {}\n pub fn count(&self) -> i64 {}\n pub fn total(&self) -> i64 {}\n}\n";
        let holes = scan_holes(code);
        assert_eq!(holes.len(), 3);
        // add -> construct-and-push a typed record from the params.
        assert!(
            holes[0].candidates.iter().any(|b| b.contains("self.items.push(Item { title: title, priority: priority });")),
            "no construct-and-push: {:?}",
            holes[0].candidates
        );
        // total -> aggregate over the record's i64 field, NOT `self.priority` (priority is Item's, not TodoList's).
        assert!(holes[2].candidates.iter().any(|b| b.contains("self.items.iter().map(|e| e.priority).sum()")), "no field-aggregate");
        assert!(holes[2].candidates.iter().all(|b| !b.contains("self.priority")), "leaked a non-self field");
    }

    #[test]
    fn scan_holes_covers_record_field_operations_with_rules() {
        let code = "pub struct Product { pub name: String, pub stock: i64 }\npub struct Shop { pub items: Vec<Product> }\nimpl Shop {\n pub fn restock(&mut self, i: i64, qty: i64) {}\n pub fn sell(&mut self, i: i64, qty: i64) {}\n}\n";
        let holes = scan_holes(code);
        assert!(holes[0].candidates.iter().any(|b| b == "self.items[i as usize].stock += qty;"), "no field +=");
        assert!(
            holes[1].candidates.iter().any(|b| b == "if qty <= self.items[i as usize].stock { self.items[i as usize].stock -= qty; }"),
            "no guarded field -="
        );
    }

    #[test]
    fn scan_holes_covers_auth_gated_mutation() {
        // A param named like a field (pin ~ self.pin) is a credential -> the mutation is access-gated.
        let code = "pub struct Account { pub balance: i64, pub pin: i64 }\nimpl Account { pub fn withdraw(&mut self, amount: i64, pin: i64) {} }\n";
        let cands = &scan_holes(code)[0].candidates;
        assert!(
            cands.iter().any(|b| b == "if pin == self.pin && amount <= self.balance { self.balance -= amount; }"),
            "no auth+overdraft guard"
        );
        // the AUTH guards never target the credential field itself (`if pin == self.pin { self.pin ...`).
        assert!(cands.iter().all(|b| !(b.starts_with("if pin ==") && b.contains("self.pin -="))), "auth guard targeted the credential field");
    }

    #[test]
    fn scan_holes_covers_guarded_mutation() {
        // A scalar-field mutator with an i64 param gets the conditional-rule guards (no-overdraft etc.).
        let code = "pub struct Account { pub balance: i64 }\nimpl Account { pub fn withdraw(&mut self, amount: i64) {} }\n";
        let cands = &scan_holes(code)[0].candidates;
        assert!(cands.iter().any(|b| b == "if amount <= self.balance { self.balance -= amount; }"), "no overdraft guard");
        assert!(cands.iter().any(|b| b == "if amount > self.balance { self.balance = amount; }"), "no keep-max guard");
    }

    #[test]
    fn scan_holes_covers_lookup_by_string_key() {
        let code = "pub struct Product { pub name: String, pub price: i64 }\npub struct Store { pub items: Vec<Product> }\nimpl Store {\n pub fn contains(&self, name: String) -> bool {}\n pub fn price_of(&self, name: String) -> i64 {}\n}\n";
        let holes = scan_holes(code);
        assert!(holes[0].candidates.iter().any(|b| b.contains("self.items.iter().any(|e| e.name == name)")), "no contains-by-key");
        assert!(
            holes[1].candidates.iter().any(|b| b.contains("self.items.iter().find(|e| e.name == name).map(|e| e.price).unwrap_or(0)")),
            "no lookup-by-key"
        );
    }

    #[test]
    fn scan_holes_covers_full_crud_over_a_typed_collection() {
        let code = "pub struct Item { pub name: String, pub price: i64 }\npub struct Cart { pub items: Vec<Item> }\nimpl Cart {\n pub fn is_empty(&self) -> bool {}\n pub fn clear(&mut self) {}\n pub fn remove_at(&mut self, i: i64) {}\n pub fn price_at(&self, i: i64) -> i64 {}\n pub fn set_price(&mut self, i: i64, p: i64) {}\n}\n";
        let holes = scan_holes(code);
        let has = |i: usize, needle: &str| holes[i].candidates.iter().any(|b| b.contains(needle));
        assert!(has(0, "self.items.is_empty()"), "is_empty");
        assert!(has(1, "self.items.clear();"), "clear");
        assert!(has(2, "self.items.remove(i as usize);"), "remove-by-index");
        assert!(has(3, "self.items[i as usize].price;"), "indexed field read");
        assert!(has(4, "self.items[i as usize].price = p;"), "field update by index");
    }

    #[test]
    fn prereq_flag_marks_pure_mutators_only() {
        // `&mut self` no-return is a prerequisite; a getter and a mutate-return are not.
        let code = "pub struct S { pub xs: Vec<i64>, pub n: i64 }\nimpl S {\n pub fn add(&mut self, x: i64) {}\n pub fn count(&self) -> i64 {}\n pub fn next(&mut self) -> i64 {}\n}\n";
        let holes = scan_holes(code);
        assert_eq!(holes.len(), 3);
        assert!(holes[0].is_prereq, "add is a prerequisite mutator");
        assert!(!holes[1].is_prereq, "count getter is not");
        assert!(!holes[2].is_prereq, "next (mutate+return) is not a pure prerequisite");
    }

    #[test]
    fn passed_count_sums_across_test_binaries() {
        let out = "test result: ok. 3 passed; 0 failed; 0 ignored\ntest result: FAILED. 2 passed; 1 failed";
        assert_eq!(passed_count(out), 5);
        assert_eq!(passed_count("no test lines here"), 0);
    }

    #[test]
    fn scan_holes_finds_multiple_holes_with_candidates_and_defaults() {
        let code = "pub struct Cart { pub prices: Vec<i64> }\nimpl Cart {\n  pub fn add(&mut self, price: i64) {}\n  pub fn count(&self) -> i64 {}\n  pub fn total(&self) -> i64 {}\n}\n";
        let holes = scan_holes(code);
        assert_eq!(holes.len(), 3, "expected add/count/total holes");
        // `add` (unit mutator) -> a push candidate + an empty default.
        assert!(holes[0].candidates.iter().any(|b| b.contains("self.prices.push(price)")), "no push for add");
        assert_eq!(holes[0].default, "", "unit default should be empty");
        // getters over the Vec field include both len and sum + an i64 default.
        assert!(holes[2].candidates.iter().any(|b| b.contains("self.prices.iter().sum()")), "no sum getter");
        assert_eq!(holes[2].default, "0");
    }

    #[test]
    fn scan_holes_gives_free_functions_param_arithmetic_not_self_fields() {
        // A free fn (no self) in a struct-bearing file must get PARAM arithmetic, never `self.X`.
        let code = "pub struct S { pub v: i64 }\npub fn net(balance: i64, fees: i64) -> i64 {}\n";
        let holes = scan_holes(code);
        let net = holes.last().expect("net hole");
        assert!(net.candidates.iter().any(|b| b.contains("balance - fees")), "no param subtraction for free fn");
        assert!(net.candidates.iter().all(|b| !b.contains("self.")), "free fn must not reference self");
    }

    #[test]
    fn mutation_field_name_swaps() {
        // `first(&self){ self.b }` should read `self.a` -- swap each self.FIELD to each other field.
        let code = "pub struct P { pub a: i64, pub b: i64 }\nimpl P { pub fn first(&self) -> i64 { self.b } }\n";
        assert!(generate_mutations(code).iter().any(|m| m.contains("{ self.a }")), "no field swap b->a");
    }

    #[test]
    fn stub_fills_generate_multistatement_and_computed_bodies() {
        // Computed getter over two fields: `total() -> self.a + self.b`.
        let g = "pub struct P { pub a: i64, pub b: i64 }\nimpl P { pub fn total(&self) -> i64 {} }\n";
        assert!(generate_stub_fills(g).iter().any(|m| m.contains("return self.a + self.b;")), "no computed getter");
        // Multi-statement mutator: `shift(dx,dy) { self.x = dx; self.y = dy; }`.
        let m = "pub struct Pt { pub x: i64, pub y: i64 }\nimpl Pt { pub fn shift(&mut self, dx: i64, dy: i64) {} }\n";
        assert!(
            generate_stub_fills(m).iter().any(|b| b.contains("self.x = dx;") && b.contains("self.y = dy;")),
            "no multi-statement mutator"
        );
        // Mutate-and-return: `next(&mut self) -> i64 { self.n += 1; return self.n; }`.
        let n = "pub struct C { pub n: i64 }\nimpl C { pub fn next(&mut self) -> i64 {} }\n";
        assert!(
            generate_stub_fills(n).iter().any(|b| b.contains("self.n += 1; return self.n;")),
            "no mutate-and-return"
        );
    }

}
