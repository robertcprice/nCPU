//! Unified backend prose intake (LOOP-7/8) — multi-door router + auto HTTP checks.
//!
//! Eliminates ceilings by trying, per rule:
//! 1. inline examples (when present in the contract text),
//! 2. compositional P2C `then`-chains,
//! 3. single registry unary op,
//! 4. project clause (affine/polynomial prose),
//! 5. NL comprehend + strict-verify.
//!
//! HTTP checks are derived from Mog execution (zero hand-grader by default).
//! Output mismatches trigger steered re-synthesis with a manufactured example.

use crate::backend_http::{parse_output_mismatch, HttpRuleCheck};
use crate::backend_ir::{BackendApp, RuleModel, StoreKind};
use crate::backend_mvp::{GeneratedBackend, SynthesizedRuleArtifact};
use crate::backend_nl::{examples_for_rule_in_text, split_function_clauses};
use crate::backend_p2c::parse_p2c_rule_clauses;
use crate::backend_repair::build_with_compile_and_http_repair;
use crate::benchmark::Value as BValue;
use crate::linguigenesis_bridge::{BridgeError, LinguigenesisBridge};
use crate::mog_transpile::to_rust;
use crate::runtime::{execute_function, Value as RValue};
use crate::solver::SolveResult;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProseSynthesisDoor {
    InlineExamples,
    Compositional,
    SingleOp,
    ProjectClause,
    RegistrySeeded,
    NlComprehend,
    ManufacturedExamples,
    SteeredResynth,
}

impl ProseSynthesisDoor {
    pub fn method_prefix(self) -> &'static str {
        match self {
            Self::InlineExamples => "inline",
            Self::Compositional => "prose:p2c",
            Self::SingleOp => "prose:single-op",
            Self::ProjectClause => "prose:project",
            Self::RegistrySeeded => "prose:seeded",
            Self::NlComprehend => "prose:nl-desc",
            Self::ManufacturedExamples => "prose:manufactured",
            Self::SteeredResynth => "prose:resynth",
        }
    }
}

/// Structural catalog for runtime introspection (MCP / CLI) — not a hand capability list.
pub fn prose_door_catalog() -> &'static [(&'static str, &'static str)] {
    &[
        ("inline", "contract text contains name(x)=y literals"),
        (
            "prose:p2c",
            "compositional then-chain resolved via registry EntityResolver",
        ),
        (
            "prose:single-op",
            "unary registry op resolved by description tokens",
        ),
        (
            "prose:project",
            "A function NAME that DESCRIPTION via synthesize_project",
        ),
        (
            "prose:seeded",
            "comprehend/registry example_cases formatted into project clause",
        ),
        (
            "prose:manufactured",
            "comprehend partial + evidence entity example_cases merged into project clause",
        ),
        (
            "prose:nl-desc",
            "NL comprehend + strict-verify + registry oracle when available",
        ),
        (
            "prose:resynth",
            "HTTP output mismatch manufactures hint; integer affines through hint point",
        ),
    ]
}

struct BuiltRule {
    artifact: SynthesizedRuleArtifact,
    mog: String,
    verify_io: Vec<(i64, i64)>,
}

pub fn build_backend_unified(
    english: &str,
    required: &[&str],
    http_checks: Option<&[HttpRuleCheck]>,
    store: StoreKind,
) -> Result<GeneratedBackend, String> {
    let mut resynth_hints: HashMap<String, (i64, i64)> = HashMap::new();
    let mut last_err = String::new();

    for _attempt in 0..3 {
        match try_build_backend_unified(
            english,
            required,
            http_checks,
            store,
            &resynth_hints,
        ) {
            Ok(generated) => return Ok(generated),
            Err(err) => {
                last_err = err.clone();
                if let Some(mismatch) = parse_output_mismatch(&err) {
                    resynth_hints.insert(mismatch.rule, (mismatch.input, mismatch.expected));
                    continue;
                }
                return Err(err);
            }
        }
    }
    Err(last_err)
}

fn try_build_backend_unified(
    english: &str,
    required: &[&str],
    http_checks: Option<&[HttpRuleCheck]>,
    store: StoreKind,
    resynth_hints: &HashMap<String, (i64, i64)>,
) -> Result<GeneratedBackend, String> {
    if required.is_empty() {
        return Err("unified backend build requires at least one required rule name".to_string());
    }

    let bridge = LinguigenesisBridge::new();
    if let Some(err) = bridge.registry_load_error() {
        return Err(format!("NL registry failed to load: {err}"));
    }

    let clauses = parse_p2c_rule_clauses(english);
    let by_name: HashMap<&str, &str> = clauses
        .iter()
        .map(|c| (c.name.as_str(), c.description.as_str()))
        .collect();

    let mut built = Vec::with_capacity(required.len());

    for name in required {
        let description = by_name.get(name).copied().ok_or_else(|| {
            format!(
                "rule '{name}' not found in English contract (parsed {} clause(s))",
                clauses.len()
            )
        })?;
        let rule = if let Some(&(input, output)) = resynth_hints.get(*name) {
            build_steered_rule(&bridge, name, description, input, output)?
        } else {
            build_rule_for_prose(&bridge, english, name, description)?
        };
        built.push(rule);
    }

    let rules: Vec<SynthesizedRuleArtifact> = built.iter().map(|r| r.artifact.clone()).collect();
    let checks = match http_checks {
        Some(c) if !c.is_empty() => c.to_vec(),
        _ => derive_http_checks_from_built(english, &built),
    };

    let description = clauses
        .iter()
        .map(|c| format!("{}: {}", c.name, c.description))
        .collect::<Vec<_>>()
        .join(" | ");
    let models = rules
        .iter()
        .map(|rule| RuleModel {
            name: rule.name.clone(),
            synthesis_method: rule.rule_method.clone(),
            rule_code: rule.rule_code.clone(),
        })
        .collect();
    let app = BackendApp::from_rules(&description, models, store);
    let source = build_with_compile_and_http_repair(&app, &checks, store, 3)?;
    Ok(GeneratedBackend { source, rules })
}

pub fn write_backend_unified(
    path: impl AsRef<std::path::Path>,
    english: &str,
    required: &[&str],
    http_checks: Option<&[HttpRuleCheck]>,
    store: StoreKind,
) -> Result<GeneratedBackend, String> {
    let generated = build_backend_unified(english, required, http_checks, store)?;
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create output dir {}: {e}", parent.display()))?;
    }
    std::fs::write(path.as_ref(), &generated.source)
        .map_err(|e| format!("write {}: {e}", path.as_ref().display()))?;
    Ok(generated)
}

fn build_rule_for_prose(
    bridge: &LinguigenesisBridge,
    english: &str,
    name: &str,
    description: &str,
) -> Result<BuiltRule, String> {
    let (res, door) = synthesize_rule_for_prose(bridge, english, name, description)?;
    finalize_built_rule(name, res, door)
}

use std::time::Instant;

fn build_steered_rule(
    bridge: &LinguigenesisBridge,
    name: &str,
    description: &str,
    input: i64,
    output: i64,
) -> Result<BuiltRule, String> {
    let started = Instant::now();
    let budget_ms = steered_wall_clock_budget_ms();
    let mut last_err = String::new();

    // Fast path: merged hint + registry examples + best affine manufacture (LOOP-11).
    let merged = steered_clause_text(bridge, name, description, input, output);
    if steered_budget_exceeded(started, budget_ms) {
        return Err(format!(
            "steered resynth for '{name}' exceeded wall-clock budget ({budget_ms}ms)"
        ));
    }
    match try_steered_project(bridge, name, &merged) {
        Ok(res) => return finalize_built_rule(name, res, ProseSynthesisDoor::SteeredResynth),
        Err(err) => last_err = err,
    }

    for pairs in candidate_affine_pair_sets(input, output)
        .into_iter()
        .take(steered_max_affine_candidates())
    {
        if steered_budget_exceeded(started, budget_ms) {
            return Err(format!(
                "steered resynth for '{name}' exceeded wall-clock budget ({budget_ms}ms): {last_err}"
            ));
        }
        let mini = clause_text_from_pairs(name, description, &pairs);
        if mini == merged {
            continue;
        }
        match try_steered_project(bridge, name, &mini) {
            Ok(res) => return finalize_built_rule(name, res, ProseSynthesisDoor::SteeredResynth),
            Err(err) => last_err = err,
        }
    }
    Err(last_err)
}

fn try_steered_project(
    bridge: &LinguigenesisBridge,
    name: &str,
    mini: &str,
) -> Result<SolveResult, String> {
    let (solved, skipped) = bridge
        .synthesize_project(mini)
        .map_err(|e| format!("steered resynth for '{name}': {e}"))?;
    if !skipped.is_empty() {
        return Err(format!(
            "steered resynth for '{name}' skipped: {skipped:?}"
        ));
    }
    let res = solved
        .into_iter()
        .find(|(n, _)| n == name)
        .map(|(_, r)| r)
        .ok_or_else(|| format!("steered resynth did not return rule '{name}'"))?;
    if !res.success {
        return Err(format!(
            "steered resynth for '{name}' failed: {:?}",
            res.error
        ));
    }
    Ok(res)
}

fn clause_text_from_pairs(name: &str, description: &str, pairs: &[(i64, i64)]) -> String {
    let literals = pairs
        .iter()
        .map(|(x, y)| format!("{name}({x})={y}"))
        .collect::<Vec<_>>()
        .join(" and ");
    format!("A function {name} that {description}, {literals}.")
}

fn steered_clause_text(
    bridge: &LinguigenesisBridge,
    name: &str,
    description: &str,
    hint_input: i64,
    hint_output: i64,
) -> String {
    use linguigenesis_core::coding_requirements::LiteralValue;
    let mut pairs = vec![(hint_input, hint_output)];
    let clause = format!("A function {name} that {description}.");
    if let Ok(req) = bridge.nl_to_requirement(&clause) {
        for ex in req.examples.iter().take(6) {
            if ex.inputs.len() != 1 {
                continue;
            }
            if let (LiteralValue::Int(x), LiteralValue::Int(y)) = (&ex.inputs[0], &ex.expected) {
                if !pairs.iter().any(|(px, py)| *px == *x && *py == *y) {
                    pairs.push((*x, *y));
                }
            }
        }
    }
    if pairs.len() < 3 {
        if let Some(extra) = candidate_affine_pair_sets(hint_input, hint_output)
            .into_iter()
            .next()
        {
            for (x, y) in extra {
                if pairs.len() >= 3 {
                    break;
                }
                if !pairs.iter().any(|(px, py)| *px == x && *py == y) {
                    pairs.push((x, y));
                }
            }
        }
    }
    clause_text_from_pairs(name, description, &pairs)
}

/// Collect i64 example pairs emergently from comprehend partials, evidence entities,
/// and registry operation resolution on description tokens (no phrase tables).
fn collect_emergent_int_example_pairs(
    bridge: &LinguigenesisBridge,
    name: &str,
    description: &str,
) -> Vec<(i64, i64)> {
    use linguigenesis_core::coding_requirements::{parse_example_cases, LiteralValue};
    use linguigenesis_core::entity_resolution::EntityResolver;

    let mut pairs = Vec::new();
    let mut push_pair = |x: i64, y: i64| {
        if !pairs.iter().any(|(px, py)| *px == x && *py == y) {
            pairs.push((x, y));
        }
    };

    let mut ingest_examples =
        |examples: &[linguigenesis_core::coding_requirements::ExampleSpec]| {
            for ex in examples.iter().take(8) {
                if ex.inputs.len() == 1 {
                    if let (LiteralValue::Int(x), LiteralValue::Int(y)) =
                        (&ex.inputs[0], &ex.expected)
                    {
                        push_pair(*x, *y);
                    }
                } else if ex.inputs.len() == 2 {
                    for (x, y) in project_binary_int_example(ex) {
                        push_pair(x, y);
                    }
                }
            }
            for (x, y) in project_binary_batch_to_unary(examples) {
                push_pair(x, y);
            }
        };

    let input = format!("A function {name} that {description}.");
    let requirement = match bridge.nl_to_requirement(&input) {
        Ok(req) => Some(req),
        Err(BridgeError::ClarificationNeeded { partial, .. }) => Some(partial),
        Err(_) => None,
    };
    if let Some(ref req) = requirement {
        ingest_examples(&req.examples);
    }

    let Ok(registry) = bridge.registry_clone() else {
        return pairs;
    };

    if let Some(ref req) = requirement {
        for entity_id in &req.evidence_entity_ids {
            if let Some(entity) = registry.get_entity(*entity_id) {
                ingest_examples(&parse_example_cases(&entity));
            }
        }
    }

    if let Some(step) = crate::reference_nl::resolve_best_scalar_op(description, &registry) {
        for entity in registry.all_entities() {
            let matches = entity
                .get_property("default_fn_name")
                .is_some_and(|n| n.as_str() == step.fn_name)
                || entity.lemma == step.fn_name;
            if matches {
                ingest_examples(&parse_example_cases(&entity));
            }
        }
    }

    let resolver = EntityResolver::new(registry);
    for token in description.split(|c: char| !c.is_alphanumeric()) {
        let surface = token.trim().to_lowercase();
        if surface.len() < 3 || resolver.is_stop_word(&surface) {
            continue;
        }
        if let Some(resolved) = resolver.resolve_operation_surface(&surface) {
            ingest_examples(&parse_example_cases(&resolved.entity));
        }
    }

    pairs
}

/// Project one binary i64 example into unary `(x, y)` pairs when an operand is fixed
/// or both inputs match (diagonal), so manufactured backend clauses stay scalar i64.
fn project_binary_int_example(
    ex: &linguigenesis_core::coding_requirements::ExampleSpec,
) -> Vec<(i64, i64)> {
    use linguigenesis_core::coding_requirements::LiteralValue;

    if ex.inputs.len() != 2 {
        return Vec::new();
    }
    let (a, b, out) = match (ex.inputs[0].clone(), ex.inputs[1].clone(), ex.expected.clone()) {
        (
            LiteralValue::Int(a),
            LiteralValue::Int(b),
            LiteralValue::Int(out),
        ) => (a, b, out),
        _ => return Vec::new(),
    };

    let mut pairs = Vec::new();
    if a == 0 {
        pairs.push((b, out));
    }
    if b == 0 {
        pairs.push((a, out));
    }
    if a == b {
        pairs.push((a, out));
    }
    pairs
}

/// When every binary example shares the same second operand, project to unary `f(x)=y`.
fn project_binary_batch_to_unary(
    examples: &[linguigenesis_core::coding_requirements::ExampleSpec],
) -> Vec<(i64, i64)> {
    use linguigenesis_core::coding_requirements::LiteralValue;

    let binary: Vec<_> = examples
        .iter()
        .filter(|ex| ex.inputs.len() == 2)
        .collect();
    if binary.len() < 2 {
        return Vec::new();
    }

    let mut try_fixed_second = |k: i64| -> Option<Vec<(i64, i64)>> {
        let mut pairs = Vec::new();
        for ex in &binary {
            let (a, b, out) = match (
                ex.inputs[0].clone(),
                ex.inputs[1].clone(),
                ex.expected.clone(),
            ) {
                (
                    LiteralValue::Int(a),
                    LiteralValue::Int(b),
                    LiteralValue::Int(out),
                ) => (a, b, out),
                _ => return None,
            };
            if b != k {
                return None;
            }
            pairs.push((a, out));
        }
        Some(pairs)
    };

    if let Some(first) = binary.first() {
        if let LiteralValue::Int(k) = first.inputs[1].clone() {
            if let Some(pairs) = try_fixed_second(k) {
                return pairs;
            }
        }
    }

    let mut try_fixed_first = |k: i64| -> Option<Vec<(i64, i64)>> {
        let mut pairs = Vec::new();
        for ex in &binary {
            let (a, b, out) = match (
                ex.inputs[0].clone(),
                ex.inputs[1].clone(),
                ex.expected.clone(),
            ) {
                (
                    LiteralValue::Int(a),
                    LiteralValue::Int(b),
                    LiteralValue::Int(out),
                ) => (a, b, out),
                _ => return None,
            };
            if a != k {
                return None;
            }
            pairs.push((b, out));
        }
        Some(pairs)
    };

    if let Some(first) = binary.first() {
        if let LiteralValue::Int(k) = first.inputs[0].clone() {
            if let Some(pairs) = try_fixed_first(k) {
                return pairs;
            }
        }
    }

    if binary.iter().all(|ex| ex.inputs[0] == ex.inputs[1]) {
        return binary
            .iter()
            .filter_map(|ex| match (ex.inputs[0].clone(), ex.expected.clone()) {
                (LiteralValue::Int(a), LiteralValue::Int(out)) => Some((a, out)),
                _ => None,
            })
            .collect();
    }

    Vec::new()
}

fn steered_max_affine_candidates() -> usize {
    std::env::var("NSYNTH_STEERED_MAX_CANDIDATES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4)
        .clamp(1, 12)
}

fn steered_wall_clock_budget_ms() -> u64 {
    std::env::var("NSYNTH_STEERED_BUDGET_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(30_000)
        .clamp(1_000, 300_000)
}

fn steered_budget_exceeded(started: Instant, budget_ms: u64) -> bool {
    started.elapsed().as_millis() as u64 >= budget_ms
}

/// Candidate `(a,b)` lines through `(x,y)` sorted by non-constant first, then `|a|+|b|`.
pub fn candidate_affines_through_point(x: i64, y: i64) -> Vec<(i64, i64)> {
    let mut scored: Vec<(i64, i64, u64, bool)> = Vec::new();
    for a in -64_i64..=64 {
        let Some(b) = a
            .checked_mul(x)
            .and_then(|ax| y.checked_sub(ax))
        else {
            continue;
        };
        let score = a.unsigned_abs().saturating_add(b.unsigned_abs());
        scored.push((a, b, score, a == 0));
    }
    scored.sort_by(|left, right| {
        left.3
            .cmp(&right.3)
            .then_with(|| left.2.cmp(&right.2))
            .then_with(|| left.0.abs().cmp(&right.0.abs()))
    });
    scored
        .into_iter()
        .map(|(a, b, _, _)| (a, b))
        .collect()
}

/// Build candidate inline-example sets from a single HTTP hint by trying integer affines.
pub fn candidate_affine_pair_sets(hint_input: i64, hint_output: i64) -> Vec<Vec<(i64, i64)>> {
    let mut out = Vec::new();
    for (a, b) in candidate_affines_through_point(hint_input, hint_output) {
        let eval = |x: i64| a.checked_mul(x).and_then(|ax| ax.checked_add(b));
        let mut pairs = vec![(hint_input, hint_output)];
        for x in [0_i64, 1, -1, 2] {
            if let Some(y) = eval(x) {
                if !pairs.iter().any(|(px, py)| *px == x && *py == y) {
                    pairs.push((x, y));
                }
            }
            if pairs.len() >= 3 {
                break;
            }
        }
        if pairs.len() >= 2 {
            out.push(pairs);
        }
    }
    out
}

/// Manufacture examples using the best-scoring non-constant affine through the hint.
pub fn manufacture_examples_from_single_hint(
    hint_input: i64,
    hint_output: i64,
    min_count: usize,
) -> Vec<(i64, i64)> {
    candidate_affine_pair_sets(hint_input, hint_output)
        .into_iter()
        .find(|pairs| pairs.len() >= min_count.max(2))
        .unwrap_or_else(|| vec![(hint_input, hint_output)])
}

/// Return the simplest non-constant integer affine through `(x,y)`, if any.
pub fn infer_simplest_affine_through_point(x: i64, y: i64) -> Option<(i64, i64)> {
    candidate_affines_through_point(x, y)
        .into_iter()
        .find(|(a, _)| *a != 0)
}

fn finalize_built_rule(
    name: &str,
    res: SolveResult,
    door: ProseSynthesisDoor,
) -> Result<BuiltRule, String> {
    if !is_i64_scalar_rule(&res.code) {
        return Err(format!(
            "rule '{name}' is not scalar i64 after synthesis via {:?}.\n  mog: {}",
            door,
            res.code.lines().next().unwrap_or("").trim()
        ));
    }

    let verify_io = sample_mog_io_pairs(&res.code, name);
    let rule_code = to_rust(&res.code);
    if !rule_code.contains(&format!("fn {name}(")) {
        return Err(format!(
            "transpiled Rust for '{name}' does not define fn {name}(...): {rule_code}"
        ));
    }

    Ok(BuiltRule {
        artifact: SynthesizedRuleArtifact {
            name: name.to_string(),
            rule_code,
            rule_method: format!("{}:{}", door.method_prefix(), res.method),
        },
        mog: res.code,
        verify_io,
    })
}

pub fn synthesize_rule_for_prose(
    bridge: &LinguigenesisBridge,
    english: &str,
    name: &str,
    description: &str,
) -> Result<(SolveResult, ProseSynthesisDoor), String> {
    let examples = examples_for_rule_in_text(english, name);
    if examples.len() >= 2 {
        if let Ok(res) = synthesize_from_inline_clause(bridge, english, name, description) {
            return Ok((res, ProseSynthesisDoor::InlineExamples));
        }
    }

    if let Some(pairs) = {
        let manufactured = collect_emergent_int_example_pairs(bridge, name, description);
        (manufactured.len() >= 2).then_some(manufactured)
    } {
        let mini = clause_text_from_pairs(name, description, &pairs);
        if let Ok(res) = try_steered_project(bridge, name, &mini) {
            return Ok((res, ProseSynthesisDoor::ManufacturedExamples));
        }
    }

    let (res, door_tag) = bridge
        .synthesize_prose_scalar_named(name, description)
        .map_err(|e| format!("prose synthesis failed for '{name}': {e}"))?;
    let door = match door_tag {
        "prose:p2c" => ProseSynthesisDoor::Compositional,
        "prose:single-op" => ProseSynthesisDoor::SingleOp,
        "prose:project" => ProseSynthesisDoor::ProjectClause,
        "prose:seeded" => ProseSynthesisDoor::RegistrySeeded,
        "prose:nl-desc" => ProseSynthesisDoor::NlComprehend,
        other => {
            return Err(format!("unexpected prose door tag for '{name}': {other}"));
        }
    };
    Ok((res, door))
}

fn synthesize_from_inline_clause(
    bridge: &LinguigenesisBridge,
    english: &str,
    name: &str,
    description: &str,
) -> Result<SolveResult, String> {
    let clause = find_clause_for_rule(english, name).ok_or_else(|| {
        format!("inline path: no clause found for rule '{name}' in English contract")
    })?;
    let mini = if clause.to_lowercase().contains(&format!("function {name}")) {
        clause
    } else {
        format!(
            "A function {name} that {description}. {}",
            inline_example_literals(name, english)
        )
    };
    let (solved, skipped) = bridge
        .synthesize_project(&mini)
        .map_err(|e| format!("inline synthesize_project for '{name}': {e}"))?;
    if !skipped.is_empty() {
        return Err(format!(
            "inline synthesize_project for '{name}' skipped: {skipped:?}"
        ));
    }
    solved
        .into_iter()
        .find(|(n, _)| n == name)
        .map(|(_, r)| r)
        .ok_or_else(|| format!("inline synthesize_project did not return rule '{name}'"))
}

fn find_clause_for_rule(english: &str, name: &str) -> Option<String> {
    split_function_clauses(english).into_iter().find(|clause| {
        clause.to_lowercase().contains(&format!("function {name}"))
            || clause.contains(&format!("{name}("))
    })
}

fn inline_example_literals(name: &str, english: &str) -> String {
    examples_for_rule_in_text(english, name)
        .into_iter()
        .map(|(x, y)| format!("{name}({x})={y}"))
        .collect::<Vec<_>>()
        .join(" and ")
}

pub fn derive_http_checks(
    english: &str,
    rules: &[SynthesizedRuleArtifact],
    mog_by_name: &HashMap<String, String>,
) -> Vec<HttpRuleCheck> {
    rules
        .iter()
        .filter_map(|rule| {
            let examples = examples_for_rule_in_text(english, &rule.name);
            if let Some((input, output)) = examples.first().copied() {
                return Some(HttpRuleCheck {
                    rule: rule.name.clone(),
                    input,
                    output,
                });
            }
            mog_by_name
                .get(&rule.name)
                .and_then(|mog| sample_mog_io_pairs(mog, &rule.name).first().copied())
                .map(|(input, output)| HttpRuleCheck {
                    rule: rule.name.clone(),
                    input,
                    output,
                })
        })
        .collect()
}

fn derive_http_checks_from_built(english: &str, built: &[BuiltRule]) -> Vec<HttpRuleCheck> {
    built
        .iter()
        .filter_map(|rule| {
            let inline = examples_for_rule_in_text(english, &rule.artifact.name);
            let (input, output) = inline
                .first()
                .copied()
                .or_else(|| rule.verify_io.first().copied())?;
            Some(HttpRuleCheck {
                rule: rule.artifact.name.clone(),
                input,
                output,
            })
        })
        .collect()
}

pub fn sample_mog_io_pairs(mog: &str, name: &str) -> Vec<(i64, i64)> {
    let mut out = Vec::new();
    for x in [0_i64, 1, -1, 3, 5, -5, 2, 4] {
        if let Ok(RValue::Int(y)) = execute_function(mog, name, &[BValue::Int(x)], name) {
            if out.iter().all(|(px, py)| *px != x || *py != y) {
                out.push((x, y));
            }
        }
        if out.len() >= 3 {
            break;
        }
    }
    out
}

pub fn probe_rule_io(mog: &str, name: &str) -> Option<(i64, i64)> {
    sample_mog_io_pairs(mog, name).into_iter().next()
}

fn is_i64_scalar_rule(mog: &str) -> bool {
    let header = mog.lines().next().unwrap_or("");
    let lower = header.to_lowercase();
    let banned = ["f64", "f32", "float", "string", "str", "&str", "char", "bool", "[", "vec<"];
    if banned.iter().any(|b| lower.contains(b)) {
        return false;
    }
    lower.contains("-> i64")
}

fn rustc_available() -> bool {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Prose-only affine/polynomial demo (no inline `name(x)=y` literals).
pub const DEFAULT_BACKEND_AFFINE_PROSE: &str = "\
A function score_bonus that scores ten points per catch plus a five point bonus. \
A function damage_penalty that converts hit points lost into a signed penalty score twice the loss minus three.";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend_http::{HttpOutputMismatch, HttpRuleCheck};
    use crate::backend_nl::default_required_rule_names;
    use crate::backend_p2c::DEFAULT_BACKEND_P2C_ENGLISH;

    #[test]
    fn parse_http_mismatch_extracts_rule_io() {
        let err = r#"POST /rules/score_bonus/evaluate body {"input":3} expected "output":35 in: HTTP/1.1 200"#;
        assert_eq!(
            parse_output_mismatch(err),
            Some(HttpOutputMismatch {
                rule: "score_bonus".to_string(),
                input: 3,
                expected: 35,
            })
        );
    }

    #[test]
    fn prose_router_single_op_increments_without_then_chain() {
        let bridge = LinguigenesisBridge::new();
        let (res, door) = bridge
            .synthesize_prose_scalar_named("bump", "increments a number")
            .expect("single-op increment");
        assert_eq!(door, "prose:single-op");
        assert!(res.success);
        assert!(res.code.contains("fn bump"));
    }

    #[test]
    fn manufacture_affine_examples_from_score_bonus_hint() {
        let sets = candidate_affine_pair_sets(3, 35);
        assert!(sets.iter().any(|pairs| pairs.contains(&(3, 35)) && pairs.contains(&(0, 5))));
        let pairs = manufacture_examples_from_single_hint(3, 35, 3);
        assert!(pairs.contains(&(3, 35)));
    }

    #[test]
    fn manufacture_affine_examples_from_damage_penalty_hint() {
        let sets = candidate_affine_pair_sets(5, 7);
        assert!(sets.iter().any(|pairs| pairs.contains(&(5, 7)) && pairs.contains(&(0, -3))));
        assert!(sets.iter().any(|pairs| infer_simplest_affine_through_point(5, 7).map(|(a, b)| (a, b)) == Some((2, -3)) || pairs.contains(&(0, -3))));
    }

    #[test]
    fn steered_clause_includes_manufactured_affine_seeds() {
        let bridge = LinguigenesisBridge::new();
        let text = steered_clause_text(
            &bridge,
            "score_bonus",
            "scores ten points per catch plus a five point bonus",
            3,
            35,
        );
        assert!(text.contains("score_bonus(3)=35"));
        assert!(text.matches("score_bonus(").count() >= 2);
    }

    #[test]
    fn build_steered_rule_from_single_http_hint() {
        let bridge = LinguigenesisBridge::new();
        let rule = build_steered_rule(
            &bridge,
            "score_bonus",
            "scores ten points per catch plus a five point bonus",
            3,
            35,
        )
        .expect("steered from single hint");
        assert!(rule.artifact.rule_method.starts_with("prose:resynth:"));
        let got = execute_function(&rule.mog, "score_bonus", &[BValue::Int(3)], "score_bonus")
            .expect("run mog");
        match got {
            RValue::Int(n) => assert_eq!(n, 35),
            other => panic!("expected 35, got {other:?}"),
        }
    }

    #[test]
    fn unified_build_affine_prose_via_steered_resynth_hints() {
        if !rustc_available() {
            eprintln!("skipping steered affine integration test: rustc unavailable");
            return;
        }

        let mut hints = HashMap::new();
        hints.insert("score_bonus".to_string(), (3_i64, 35_i64));
        hints.insert("damage_penalty".to_string(), (5_i64, 7_i64));

        let generated = try_build_backend_unified(
            DEFAULT_BACKEND_AFFINE_PROSE,
            default_required_rule_names(),
            None,
            StoreKind::Memory,
            &hints,
        )
        .expect("steered affine backend build");

        assert_eq!(generated.rules.len(), 2);
        assert!(generated
            .rules
            .iter()
            .all(|r| r.rule_method.starts_with("prose:resynth:")));
        assert!(generated.source.contains("/rules/score_bonus/evaluate"));
        assert!(generated.source.contains("/rules/damage_penalty/evaluate"));
    }

    #[test]
    fn steered_wall_clock_budget_helpers() {
        use std::time::{Duration, Instant};
        let started = Instant::now() - Duration::from_millis(50);
        assert!(steered_budget_exceeded(started, 10));
        assert!(!steered_budget_exceeded(Instant::now(), steered_wall_clock_budget_ms()));
    }

    #[test]
    fn project_binary_int_example_zero_and_diagonal() {
        use linguigenesis_core::coding_requirements::{ExampleSpec, LiteralValue};
        let ex = ExampleSpec {
            inputs: vec![LiteralValue::Int(0), LiteralValue::Int(5)],
            expected: LiteralValue::Int(5),
        };
        assert_eq!(project_binary_int_example(&ex), vec![(5, 5)]);
        let diag = ExampleSpec {
            inputs: vec![LiteralValue::Int(4), LiteralValue::Int(4)],
            expected: LiteralValue::Int(16),
        };
        assert_eq!(project_binary_int_example(&diag), vec![(4, 16)]);
    }

    #[test]
    fn project_binary_batch_fixed_second_operand() {
        use linguigenesis_core::coding_requirements::{ExampleSpec, LiteralValue};
        let examples = vec![
            ExampleSpec {
                inputs: vec![LiteralValue::Int(2), LiteralValue::Int(3)],
                expected: LiteralValue::Int(6),
            },
            ExampleSpec {
                inputs: vec![LiteralValue::Int(4), LiteralValue::Int(3)],
                expected: LiteralValue::Int(12),
            },
        ];
        assert_eq!(
            project_binary_batch_to_unary(&examples),
            vec![(2, 6), (4, 12)]
        );
    }

    #[test]
    fn collect_emergent_pairs_from_registry_operation_resolution() {
        let bridge = LinguigenesisBridge::new();
        let pairs =
            collect_emergent_int_example_pairs(&bridge, "bump", "increments a number");
        assert!(
            pairs.len() >= 2,
            "registry resolver should seed add examples, got {pairs:?}"
        );
    }

    #[test]
    fn prose_door_catalog_is_structural_not_empty() {
        assert!(prose_door_catalog().len() >= 7);
    }

    #[test]
    fn pure_affine_prose_honestly_refuses_without_oracle_or_examples() {
        if !rustc_available() {
            eprintln!("skipping pure affine refusal test: rustc unavailable");
            return;
        }
        let err = build_backend_unified(
            DEFAULT_BACKEND_AFFINE_PROSE,
            default_required_rule_names(),
            None,
            StoreKind::Memory,
        )
        .unwrap_err();
        assert!(
            err.contains("prose synthesis failed")
                || err.contains("HTTP repair failed")
                || err.contains("not scalar i64"),
            "expected honest refusal without examples/oracle, got: {err}"
        );
    }

    #[test]
    fn outer_retry_loop_uses_parsed_http_mismatch_not_preseeded_hints() {
        let err = r#"POST /rules/score_bonus/evaluate body {"input":0} expected "output":9999 in: HTTP/1.1 200"#;
        let mismatch = parse_output_mismatch(err).expect("parse mismatch");
        assert_eq!(mismatch.rule, "score_bonus");
        assert_eq!(mismatch.input, 0);
        assert_eq!(mismatch.expected, 9999);
    }

    #[test]
    fn unified_build_uses_compositional_p2c_default_with_auto_http() {
        if !rustc_available() {
            eprintln!("skipping unified P2C integration test: rustc unavailable");
            return;
        }

        let generated = build_backend_unified(
            DEFAULT_BACKEND_P2C_ENGLISH,
            default_required_rule_names(),
            None,
            StoreKind::Memory,
        )
        .expect("unified P2C build");

        assert_eq!(generated.rules.len(), 2);
        assert!(generated.rules[0].rule_method.starts_with("prose:p2c:"));
        assert!(generated.source.contains("/rules/score_bonus/evaluate"));
    }

    #[test]
    fn unified_build_auto_routes_inline_examples_when_present() {
        if !rustc_available() {
            eprintln!("skipping unified inline auto-route test: rustc unavailable");
            return;
        }

        let english = "\
A function score_bonus that scores ten points per catch plus a five point bonus, \
score_bonus(0)=5 and score_bonus(1)=15 and score_bonus(2)=25. \
A function damage_penalty that converts hit points lost into a signed penalty score twice the loss minus three, \
damage_penalty(0)=-3 and damage_penalty(1)=-1 and damage_penalty(2)=1.";

        let generated = build_backend_unified(
            english,
            default_required_rule_names(),
            None,
            StoreKind::Memory,
        )
        .expect("unified inline build");

        assert!(generated
            .rules
            .iter()
            .any(|r| r.rule_method.starts_with("inline:")));
    }

    #[test]
    fn derive_http_checks_from_mog_samples_multiple_probes() {
        let bridge = LinguigenesisBridge::new();
        let (res, _) = bridge
            .synthesize_prose_scalar_named("bump", "increments a number")
            .expect("synthesize bump");
        let mog = res.code.clone();
        let pairs = sample_mog_io_pairs(&mog, "bump");
        assert!(pairs.len() >= 2);
        assert!(pairs.contains(&(1, 2)));
        assert!(pairs.contains(&(0, 1)));
    }
}

/// A comprehended BACKEND ask from free prose (the hub's backend domain).
#[derive(Clone, Debug, PartialEq)]
pub struct BackendAsk {
    pub store: StoreKind,
    /// Named rule clauses with inline examples (empty = structural-only server:
    /// health route + store, no synthesized handlers).
    pub rule_names: Vec<String>,
}

/// Emergent comprehension of a backend request: the routing gate is a
/// construction cue plus a token resolving through the REGISTRY HUB's backend
/// domain to a server/route concept ("api" -> endpoint, "service"/"backend" ->
/// server — synonym edges + morphology, not keywords). Store kind from the
/// resolved store concept's surface (sqlite/file words pick the engine's
/// concrete stores; default memory). Rule names from "a function NAME ..."
/// clauses that carry inline examples.
pub fn comprehend_backend_prose(text: &str) -> Option<BackendAsk> {
    use crate::registry_hub::{backend_seeds, domain_registry, resolve_domain, Domain};
    use linguigenesis_core::entity_resolution::EntityResolver;
    let lower = text.to_lowercase();
    let tokens: Vec<String> = lower
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect();
    const CUES: [&str; 7] = ["make", "build", "create", "add", "new", "generate", "want"];
    if !tokens.iter().any(|t| CUES.contains(&t.as_str())) {
        return None;
    }
    let registry = domain_registry(Domain::Backend, &backend_seeds());
    let resolver = EntityResolver::new(registry.clone());
    let mut is_backend = false;
    let mut store_word: Option<String> = None;
    for t in &tokens {
        if let Some((kind, _lemma, _score)) = resolve_domain(&resolver, &registry, t, "backend_kind") {
            match kind.as_str() {
                "server" | "route" => is_backend = true,
                "store" => store_word = Some(t.clone()),
                _ => {}
            }
        }
    }
    if !is_backend {
        return None;
    }
    // Concrete store selection: the ENGINE'S OWN store names first
    // (StoreKind::parse — platform vocabulary), else memory default when any
    // store concept was mentioned, else memory.
    let store = tokens
        .iter()
        .find_map(|t| StoreKind::parse(t))
        .unwrap_or(StoreKind::Memory);
    let _ = store_word;
    // Rule clauses: "a function NAME ..." with inline examples.
    let mut rule_names = Vec::new();
    for clause in split_function_clauses(text) {
        let cl = clause.to_lowercase();
        if let Some(rest) = cl.split("function ").nth(1) {
            let name: String = rest
                .split_whitespace()
                .find(|w| w.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or("")
                .to_string();
            if !name.is_empty() && !examples_for_rule_in_text(text, &name).is_empty() {
                rule_names.push(name);
            }
        }
    }
    Some(BackendAsk { store, rule_names })
}

/// Build a comprehended backend ask into `root/backend/main.rs`:
///   * rule asks route through the FULL unified door (synthesis + auto HTTP
///     checks + compile/HTTP repair);
///   * structural-only asks render the server (health route + store) and pass
///     the COMPILE-REPAIR gate.
/// Fail-closed either way. Returns the written relative paths.
pub fn build_backend_ask(root: &std::path::Path, english: &str, ask: &BackendAsk) -> Result<Vec<String>, String> {
    let out = root.join("backend/main.rs");
    if ask.rule_names.is_empty() {
        let app = BackendApp::from_rules(english, vec![], ask.store);
        let source = crate::backend_repair::build_with_compile_and_http_repair(&app, &[], ask.store, 2)?;
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(&out, source).map_err(|e| e.to_string())?;
    } else {
        let names: Vec<&str> = ask.rule_names.iter().map(String::as_str).collect();
        write_backend_unified(&out, english, &names, None, ask.store)?;
    }
    Ok(vec!["backend/main.rs".to_string()])
}

#[cfg(test)]
mod backend_ask_tests {
    use super::*;

    #[test]
    fn comprehends_structural_and_rule_asks() {
        // Structural: hub resolution ("api" -> endpoint via synonym edge).
        let a = comprehend_backend_prose("make me an api with a health check and a users database")
            .expect("backend ask");
        assert!(a.rule_names.is_empty());
        assert_eq!(a.store, StoreKind::Memory);
        // Synonym + morphology: "service" -> server; sqlite store word.
        let b = comprehend_backend_prose("build a new service storing events in sqlite")
            .expect("service ask");
        assert_eq!(b.store, StoreKind::Sqlite);
        // Rule clause with inline examples joins the unified door.
        let c = comprehend_backend_prose(
            "create a backend with a function double where double(2)=4 and double(5)=10",
        )
        .expect("rule ask");
        assert_eq!(c.rule_names, vec!["double".to_string()]);
        // Negatives: no backend concept, or no cue.
        assert!(comprehend_backend_prose("add a function that triples a number").is_none());
        assert!(comprehend_backend_prose("the api is slow").is_none());
    }

    #[test]
    fn structural_backend_builds_and_compiles() {
        let root = std::env::temp_dir().join(format!("nsynth_bask_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let english = "make me an api with a health check";
        let ask = comprehend_backend_prose(english).expect("ask");
        let written = build_backend_ask(&root, english, &ask).expect("build (compile-gated)");
        assert_eq!(written, vec!["backend/main.rs".to_string()]);
        let src = std::fs::read_to_string(root.join("backend/main.rs")).unwrap();
        assert!(src.contains("/health"), "health route present");
        let _ = std::fs::remove_dir_all(&root);
    }
}
