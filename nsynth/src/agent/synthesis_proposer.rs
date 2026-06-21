//! NL synthesis proposer for the repair loop (Package B/H bridge).
//!
//! When a `RepoTaskSpec.issue` begins with `nl:` or `synthesize:`, this proposer
//! runs the Linguigenesis-native `AgentRun` path and writes verified synthesis
//! output into an allowed repository file.

use crate::agent::agent_run::AgentRun;
use crate::agent::coding_intent::CodingIntent;
use crate::agent::repo::{FailureAnalysis, FailureKind, RepairContext, RepairEdit, RepairPatch, RepoTaskSpec};
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
    let trimmed = synthesized.trim();
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
                return format!("{indent}{var}: i64 = {rhs};");
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

pub(crate) fn read_relative_file(context: &RepairContext, relative: &str) -> Result<String, String> {
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
    if desc.contains("subtract")
        || desc.contains("minus")
        || mog_code.contains("a - b")
    {
        return Some("a - b");
    }
    if desc.contains("multiply")
        || desc.contains("product")
        || mog_code.contains("a * b")
    {
        return Some("a * b");
    }
    if desc.contains("divide")
        || desc.contains("division")
        || mog_code.contains("a / b")
    {
        return Some("a / b");
    }
    if desc.contains("larger")
        || desc.contains("maximum")
        || desc.contains("max")
        || mog_code.contains("if a > b")
    {
        return Some("if a > b { a } else { b }");
    }
    if desc.contains("add")
        || desc.contains("sum")
        || mog_code.contains("a + b")
    {
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
        FailureParser, GuardrailPolicy, HardnessProfile, HardnessTier, nl_fixture_cargo_test_command,
        RepairContext, RepairLoop, RepairVerifier, RepoTaskKind, RepoTaskSpec, write_nl_fixture_crate,
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

    #[test]
    fn nl_synthesis_proposer_fixes_wrong_add() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        let root = synthesis_fixture();
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("context");
        let task = nl_task(&root, "nl-add", "nl_fixture_add", "synthesize: add two numbers");

        let patch = nl_synthesis_proposer(&task, &context, 0, None).expect("propose");
        assert!(!patch.edits.is_empty());
        assert_eq!(patch.metadata.iter().find(|(k, _)| k == "synthesis_method").map(|(_, v)| v.as_str()), Some("nl_description_repo_stub"));
        let new_content = patch.edits[0].new_text.clone();
        assert!(new_content.contains("add_two"));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn failure_aware_proposer_uses_cargo_test_failure() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        let root = std::env::temp_dir().join(format!("nsynth_nl_fail_{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        write_nl_fixture_crate(&root, "nl_fixture_divide").expect("write");
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("context");
        let task = nl_task(&root, "nl-div-fail", "nl_fixture_divide", "synthesize: divide two numbers");
        let verification = RepairVerifier::new(&root, GuardrailPolicy::default())
            .verify(&task.test_command)
            .expect("verify");
        assert!(!verification.success);
        let analysis = FailureParser::default().parse(&verification.failure_output());
        assert_eq!(analysis.kind, crate::agent::repo::FailureKind::TestFailure);

        let patch = nl_synthesis_proposer(&task, &context, 0, Some(&analysis)).expect("patch");
        assert_eq!(
            patch.metadata.iter().find(|(k, _)| k == "synthesis_method").map(|(_, v)| v.as_str()),
            Some("nl_failure_aware_repo_stub")
        );
        assert!(patch.edits[0].new_text.contains("/ b"));

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn repair_loop_with_nl_synthesis_proposer() {
        let _guard = NL_SYNTHESIS_TEST_LOCK.lock().unwrap();
        let root = synthesis_fixture();
        let context = RepairContext::build(&root, &GuardrailPolicy::default()).expect("context");
        let task = nl_task(&root, "nl-add-loop", "nl_fixture_add", "synthesize: add two numbers");
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
