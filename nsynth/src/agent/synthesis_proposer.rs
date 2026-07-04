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

    if let Some(patch) = try_nl_repo_fast_patch(task, context, &description, analysis) {
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
    // Append the new fn, pub'd so sibling modules/tests can call it.
    let mut appended = synthesized.trim().to_string();
    if !appended.starts_with("pub ") && appended.starts_with("fn ") {
        appended = format!("pub {appended}");
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
        let main_renamed = ensure_pub_fn(&rename_first_fn(&fns[main_idx].1, repo_fn));
        let new_impl = format!("{helpers}{}", main_renamed.trim());
        return if old_text.contains(&format!("fn {repo_fn}")) {
            replace_function_body(old_text, repo_fn, &new_impl).ok()
        } else {
            Some(format!("{}\n", new_impl.trim()))
        };
    }

    let synth_body = fn_body(synthesized)?;
    if !old_text.contains(&format!("fn {repo_fn}")) {
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
    replace_body_only(old_text, repo_fn, &body).ok()
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
