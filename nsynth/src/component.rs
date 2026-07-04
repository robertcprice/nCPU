//! Component layer: named, NL-resolvable, COMPOSABLE, VERIFIED units — one level
//! up from single ops.
//!
//! An op is a single verified function (`array_sum`). A COMPONENT is a named unit
//! resolved from ONE natural-language phrase. Two shapes:
//!
//!   * **Bundle** — several verified leaf ops composed into one compile-gated
//!     module (`array_stats` = sum/max/min/average/length).
//!   * **Structural** — a bundle PLUS a raw-Rust glue module (a struct + methods)
//!     whose bodies call the verified leaves (`Counter` = a struct whose `tick`
//!     uses the verified `increment` op). This is the genuine capability lift: the
//!     greenfield writer alone cannot emit structs; a structural component pairs a
//!     hand-glue *shape* with synthesized+verified *logic*, exactly the game/backend
//!     builder pattern generalized into a reusable unit.
//!
//! Every leaf keeps the engine's 0-false-positive guarantee; the WHOLE assembly
//! (leaves + struct glue) is verified by the same `cargo check` gate the greenfield
//! writer uses. Nothing is trusted-but-unverified — a struct that references a leaf
//! that didn't synthesize, or glue that mis-types, fails compilation and is caught.
//!
//! FIRST SLICES done: bundle + structural + literal resolution. Extends to emergent
//! NL resolution (reusing the op resolver) and a DATA registry grown by mining
//! verified builds ("writes its own teachers" at the component grain).

use crate::agent::repo::nl_fixture_harness::{
    behavior_gate, compile_gate, write_synthesized_project, CompileStatus, WriteOutcome,
};
use crate::linguigenesis_bridge::LinguigenesisBridge;
use linguigenesis_core::entity_resolution::{edit_distance, morphological_variants};
use serde::Deserialize;
use std::path::Path;
use std::sync::OnceLock;

/// A raw-Rust glue module (struct + methods) whose bodies call the component's
/// verified leaves. Written verbatim next to the transpiled leaves and wired into
/// `lib.rs`, then compile-gated with them. Owned (not `&'static`) so components can
/// be loaded from DATA, not only baked into the binary.
#[derive(Clone, Deserialize)]
pub struct GlueSpec {
    /// Glue module name (`src/<module>.rs`).
    pub module: String,
    /// Raw Rust: a struct + impl that `use`s the leaf functions (each leaf `foo`
    /// is available as `crate::foo::foo`).
    pub code: String,
    /// Optional behavioral contract: a raw-Rust `#[cfg(test)]` module that
    /// constructs the struct, exercises its methods, and ASSERTS runtime output.
    /// Appended to the glue module and run with `cargo test` — the rung above
    /// compilation. A struct that type-checks but whose synthesized logic
    /// misbehaves (e.g. `increment` that didn't actually add 1) fails here.
    #[serde(default)]
    pub smoke: Option<String>,
}

/// A named unit bigger than a single op. Owned + `Deserialize` so the registry can
/// be grown from a JSON data file (like `coding_registry.json`), not just the seeds.
#[derive(Clone, Deserialize)]
pub struct ComponentSpec {
    /// Module + package name for the emitted component.
    pub name: String,
    /// Natural-language surface words that resolve to this component.
    pub surfaces: Vec<String>,
    /// `default_fn_name`s of the leaf ops this component bundles. Each is
    /// independently verified-synthesizable via the trusted op path.
    pub leaves: Vec<String>,
    /// Optional struct/method glue over the leaves (structural component).
    #[serde(default)]
    pub glue: Option<GlueSpec>,
}

/// The built-in SEED components — the baseline that ships in the binary. The live
/// registry is `seed_components()` merged with any DATA-defined components (by
/// name), so seeds can be extended or overridden without a recompile.
fn seed_components() -> Vec<ComponentSpec> {
    let s = |xs: &[&str]| xs.iter().map(|x| x.to_string()).collect::<Vec<String>>();
    vec![
        ComponentSpec {
            name: "array_stats".into(),
            surfaces: s(&["stats", "statistics", "statistic", "summary"]),
            leaves: s(&["array_sum", "array_max", "array_min", "average", "length"]),
            glue: None,
        },
        ComponentSpec {
            name: "counter".into(),
            surfaces: s(&["counter", "count", "tally"]),
            leaves: s(&["increment"]),
            glue: Some(GlueSpec {
                module: "counter".into(),
                code: COUNTER_GLUE.into(),
                smoke: Some(COUNTER_SMOKE.into()),
            }),
        },
        ComponentSpec {
            name: "accumulator".into(),
            surfaces: s(&["accumulator", "accumulate", "accumulation"]),
            leaves: s(&["add"]),
            glue: Some(GlueSpec {
                module: "accumulator".into(),
                code: ACCUMULATOR_GLUE.into(),
                smoke: Some(ACCUMULATOR_SMOKE.into()),
            }),
        },
        ComponentSpec {
            name: "scaler".into(),
            surfaces: s(&["scaler", "scale", "scaling", "multiplier"]),
            leaves: s(&["multiply"]),
            glue: Some(GlueSpec {
                module: "scaler".into(),
                code: SCALER_GLUE.into(),
                smoke: Some(SCALER_SMOKE.into()),
            }),
        },
        ComponentSpec {
            name: "aggregator".into(),
            surfaces: s(&["aggregator", "aggregate", "aggregation"]),
            leaves: s(&["array_sum", "array_max", "array_min", "length"]),
            glue: Some(GlueSpec {
                module: "aggregator".into(),
                code: AGGREGATOR_GLUE.into(),
                smoke: Some(AGGREGATOR_SMOKE.into()),
            }),
        },
        ComponentSpec {
            name: "stack".into(),
            surfaces: s(&["stack", "lifo"]),
            leaves: s(&["length"]),
            glue: Some(GlueSpec {
                module: "stack".into(),
                code: STACK_GLUE.into(),
                smoke: Some(STACK_SMOKE.into()),
            }),
        },
        ComponentSpec {
            name: "queue".into(),
            surfaces: s(&["queue", "fifo"]),
            leaves: s(&["length"]),
            glue: Some(GlueSpec {
                module: "queue".into(),
                code: QUEUE_GLUE.into(),
                smoke: Some(QUEUE_SMOKE.into()),
            }),
        },
    ]
}

/// Parse a components JSON document (`[{name, surfaces, leaves, glue?}, ...]`) into
/// specs. Pure — the unit of DATA extensibility, tested directly without touching
/// the process-wide cached registry.
pub fn parse_components_json(text: &str) -> Result<Vec<ComponentSpec>, String> {
    serde_json::from_str::<Vec<ComponentSpec>>(text).map_err(|e| e.to_string())
}

/// Load DATA-defined components from a JSON file: `NSYNTH_COMPONENTS` env path if
/// set, else a couple of conventional locations. Returns `None` when absent or
/// unparseable (the seeds are always the floor — a bad data file never breaks the
/// built-ins).
fn load_data_components() -> Option<Vec<ComponentSpec>> {
    let candidates: Vec<std::path::PathBuf> = std::env::var("NSYNTH_COMPONENTS")
        .ok()
        .map(std::path::PathBuf::from)
        .into_iter()
        .chain(
            [
                "data/components.json",
                "../linguigenesis/data/components.json",
                "../../linguigenesis/data/components.json",
            ]
            .iter()
            .map(std::path::PathBuf::from),
        )
        .collect();
    for path in candidates {
        if let Ok(text) = std::fs::read_to_string(&path) {
            if let Ok(specs) = parse_components_json(&text) {
                return Some(specs);
            }
        }
    }
    None
}

/// The live component registry: seeds merged with DATA-defined components (merge by
/// name — data overrides a seed, new names extend). Cached once per process; returns
/// a `'static` slice so every accessor keeps returning `&'static ComponentSpec`.
pub fn registry() -> &'static [ComponentSpec] {
    static REG: OnceLock<Vec<ComponentSpec>> = OnceLock::new();
    REG.get_or_init(|| {
        let mut comps = seed_components();
        if let Some(extra) = load_data_components() {
            for c in extra {
                if let Some(slot) = comps.iter_mut().find(|x| x.name == c.name) {
                    *slot = c;
                } else {
                    comps.push(c);
                }
            }
        }
        comps
    })
    .as_slice()
}

/// Behavioral contract for `Counter`: three ticks must land on 3. This asserts the
/// SYNTHESIZED `increment` genuinely adds 1 each call — runtime proof, not types.
const COUNTER_SMOKE: &str = r#"
#[cfg(test)]
mod counter_behaves {
    use super::Counter;
    // PROPERTY: after k ticks the count is exactly k, for all k in 1..=100 — proves
    // the synthesized `increment` is +1 across a range, not just one example.
    #[test]
    fn every_tick_increments_by_one() {
        let mut c = Counter::new();
        for k in 1..=100i64 {
            c.tick();
            assert_eq!(c.get(), k, "after {k} ticks");
        }
    }
}
"#;

/// A running total whose `accumulate(x)` folds `x` in via the VERIFIED 2-arg `add`
/// leaf. Proves the structural pattern generalizes past a nullary tick: the method
/// takes an ARGUMENT and the backing leaf is binary.
const ACCUMULATOR_GLUE: &str = r#"//! Structural component: an Accumulator that folds values via the verified `add` leaf.

use crate::add::add;

#[derive(Default)]
pub struct Accumulator {
    total: i64,
}

impl Accumulator {
    pub fn new() -> Self {
        Accumulator { total: 0 }
    }
    /// Fold a value into the running total using the synthesized + verified `add` op.
    pub fn accumulate(&mut self, x: i64) {
        self.total = add(self.total, x);
    }
    pub fn total(&self) -> i64 {
        self.total
    }
}
"#;

/// Behavioral contract for `Accumulator`: 5 + 3 + 10 must total 18 — proving the
/// synthesized `add` genuinely sums its two arguments.
const ACCUMULATOR_SMOKE: &str = r#"
#[cfg(test)]
mod accumulator_behaves {
    use super::Accumulator;
    // PROPERTY (differential): the running total equals the native running sum
    // across a varied sequence incl. negatives and large magnitudes — proves the
    // synthesized `add` matches `+` beyond one example.
    #[test]
    fn total_matches_native_running_sum() {
        let mut a = Accumulator::new();
        let mut expected = 0i64;
        for x in [5, 3, 10, -2, 0, 100, -50, 999, -1000, 7] {
            a.accumulate(x);
            expected += x;
            assert_eq!(a.total(), expected, "after adding {x}");
        }
    }
}
"#;

/// A scaler whose factor is set at CONSTRUCTION and applied via the VERIFIED 2-arg
/// `multiply` leaf. A third structural SHAPE: state is a configured operand (not an
/// accumulator), and the constructor takes an argument.
const SCALER_GLUE: &str = r#"//! Structural component: a Scaler that multiplies by a configured factor via the verified `multiply` leaf.

use crate::multiply::multiply;

pub struct Scaler {
    factor: i64,
}

impl Scaler {
    pub fn new(factor: i64) -> Self {
        Scaler { factor }
    }
    /// Scale a value by the configured factor using the synthesized + verified `multiply` op.
    pub fn scale(&self, x: i64) -> i64 {
        multiply(x, self.factor)
    }
}
"#;

/// Behavioral contract for `Scaler`: factor 3 scales 5 -> 15 and 0 -> 0, proving
/// the synthesized `multiply` genuinely multiplies its two arguments.
const SCALER_SMOKE: &str = r#"
#[cfg(test)]
mod scaler_behaves {
    use super::Scaler;
    // PROPERTY (differential): scale(x) equals f*x across a grid of factors and
    // inputs (incl. negatives and zero) — proves the synthesized `multiply` matches
    // `*` beyond one example.
    #[test]
    fn scale_matches_native_product() {
        for f in [-3i64, 0, 1, 3, 7, 50] {
            let s = Scaler::new(f);
            for x in [-4i64, 0, 1, 5, 9, 123] {
                assert_eq!(s.scale(x), f * x, "factor {f} times {x}");
            }
        }
    }
}
"#;

/// A collection-state component: the state is a Vec (not a scalar), `push` is
/// trivial std plumbing, and the QUERY methods delegate to the VERIFIED array
/// reducers. `mean` composes two verified leaves (array_sum / length). This is the
/// fourth structural SHAPE — aggregate state answered by synthesized logic.
const AGGREGATOR_GLUE: &str = r#"//! Structural component: an Aggregator over a Vec, answered by the verified array reducers.

use crate::array_sum::array_sum;
use crate::array_max::array_max;
use crate::array_min::array_min;
use crate::length::length;

#[derive(Default)]
pub struct Aggregator {
    data: Vec<i64>,
}

impl Aggregator {
    pub fn new() -> Self {
        Aggregator { data: Vec::new() }
    }
    /// Trivial std plumbing — the interesting logic is in the verified reducers.
    pub fn push(&mut self, x: i64) {
        self.data.push(x);
    }
    pub fn sum(&self) -> i64 {
        array_sum(self.data.clone())
    }
    pub fn max(&self) -> i64 {
        array_max(self.data.clone())
    }
    pub fn min(&self) -> i64 {
        array_min(self.data.clone())
    }
    pub fn count(&self) -> i64 {
        length(self.data.clone())
    }
    /// Integer mean = sum / count, both VERIFIED leaves; 0 on empty.
    pub fn mean(&self) -> i64 {
        let n = length(self.data.clone());
        if n == 0 {
            0
        } else {
            array_sum(self.data.clone()) / n
        }
    }
}
"#;

/// Behavioral contract for `Aggregator`: over [2,4,6,8] sum=20, max=8, min=2,
/// count=4, mean=5 — proving the synthesized reducers answer aggregate state.
const AGGREGATOR_SMOKE: &str = r#"
#[cfg(test)]
mod aggregator_behaves {
    use super::Aggregator;
    // PROPERTY (differential): every query equals the native Rust reduction over
    // the same inputs — proves the synthesized reducers match std across an
    // unsorted set with duplicates and negatives, not one hand-picked case.
    #[test]
    fn queries_match_native_reductions() {
        let inputs = [7i64, 2, 9, 4, 2, 100, -3, 50, 0, 9];
        let mut a = Aggregator::new();
        for &x in inputs.iter() {
            a.push(x);
        }
        let native_sum: i64 = inputs.iter().sum();
        let native_max = *inputs.iter().max().unwrap();
        let native_min = *inputs.iter().min().unwrap();
        let n = inputs.len() as i64;
        assert_eq!(a.sum(), native_sum);
        assert_eq!(a.max(), native_max);
        assert_eq!(a.min(), native_min);
        assert_eq!(a.count(), n);
        assert_eq!(a.mean(), native_sum / n);
    }
}
"#;

/// A LIFO Stack. push/pop are std plumbing (correct by construction); the LIFO
/// ORDER is proven behaviorally by the contract, and `size` uses the verified
/// `length` leaf. Collection-state shape with data-structure semantics.
const STACK_GLUE: &str = r#"//! Structural component: a LIFO Stack; order behaviorally verified, size via the verified `length` leaf.

use crate::length::length;

#[derive(Default)]
pub struct Stack {
    items: Vec<i64>,
}

impl Stack {
    pub fn new() -> Self {
        Stack { items: Vec::new() }
    }
    pub fn push(&mut self, x: i64) {
        self.items.push(x);
    }
    pub fn pop(&mut self) -> Option<i64> {
        self.items.pop()
    }
    /// Size via the verified `length` leaf.
    pub fn size(&self) -> i64 {
        length(self.items.clone())
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}
"#;

/// Behavioral contract for `Stack`: pop returns reverse insertion order (LIFO) and
/// `size` (verified `length` leaf) tracks the count.
const STACK_SMOKE: &str = r#"
#[cfg(test)]
mod stack_behaves {
    use super::Stack;
    #[test]
    fn lifo_order_and_verified_size() {
        let mut s = Stack::new();
        assert!(s.is_empty());
        for x in [1, 2, 3, 4, 5] {
            s.push(x);
        }
        assert_eq!(s.size(), 5);
        let mut got = Vec::new();
        while let Some(v) = s.pop() {
            got.push(v);
        }
        assert_eq!(got, vec![5, 4, 3, 2, 1], "LIFO");
        assert_eq!(s.size(), 0);
        assert!(s.is_empty());
    }
}
"#;

/// A FIFO Queue over a VecDeque. enqueue/dequeue are std plumbing; the FIFO ORDER
/// is proven behaviorally, `size` uses the verified `length` leaf.
const QUEUE_GLUE: &str = r#"//! Structural component: a FIFO Queue; order behaviorally verified, size via the verified `length` leaf.

use std::collections::VecDeque;
use crate::length::length;

#[derive(Default)]
pub struct Queue {
    items: VecDeque<i64>,
}

impl Queue {
    pub fn new() -> Self {
        Queue { items: VecDeque::new() }
    }
    pub fn enqueue(&mut self, x: i64) {
        self.items.push_back(x);
    }
    pub fn dequeue(&mut self) -> Option<i64> {
        self.items.pop_front()
    }
    /// Size via the verified `length` leaf.
    pub fn size(&self) -> i64 {
        length(self.items.iter().copied().collect::<Vec<i64>>())
    }
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}
"#;

/// Behavioral contract for `Queue`: dequeue returns insertion order (FIFO) and
/// `size` (verified `length` leaf) tracks the count.
const QUEUE_SMOKE: &str = r#"
#[cfg(test)]
mod queue_behaves {
    use super::Queue;
    #[test]
    fn fifo_order_and_verified_size() {
        let mut q = Queue::new();
        for x in [1, 2, 3, 4, 5] {
            q.enqueue(x);
        }
        assert_eq!(q.size(), 5);
        let mut got = Vec::new();
        while let Some(v) = q.dequeue() {
            got.push(v);
        }
        assert_eq!(got, vec![1, 2, 3, 4, 5], "FIFO");
        assert_eq!(q.size(), 0);
    }
}
"#;

/// A counter whose `tick` uses the VERIFIED `increment` leaf (x -> x+1). The struct
/// SHAPE is templated; the increment LOGIC is synthesized + verified; the whole
/// compiles together or is rejected.
const COUNTER_GLUE: &str = r#"//! Structural component: a Counter whose tick uses the verified `increment` leaf.

use crate::increment::increment;

#[derive(Default)]
pub struct Counter {
    count: i64,
}

impl Counter {
    pub fn new() -> Self {
        Counter { count: 0 }
    }
    /// Advance the counter using the synthesized + verified `increment` op.
    pub fn tick(&mut self) {
        self.count = increment(self.count);
    }
    pub fn get(&self) -> i64 {
        self.count
    }
}
"#;

/// Match tiers, strongest first. A surface is a minimal SEED; recognition
/// generalizes emergently off it (morphology + tight fuzzy), the same seed-plus-
/// emergent pattern the op registry uses — NOT a hand-maintained synonym list.
const TIER_EXACT: u8 = 3;
const TIER_MORPH: u8 = 2;
const TIER_FUZZY: u8 = 1;

/// Emergent match of one phrase `token` against one seed `surface`:
///   * exact           — token == surface
///   * morphological   — a shared stem (strip -ing/-ed/-s/-es/-ly, both sides), so
///                       "counting"/"counters"/"tallying" reach count/counter/tally
///     with no per-inflection entry
///   * fuzzy           — edit distance <= 1 on words >= 5 chars (typo tolerance),
///                       conservative so "count" never leaks into "mount"/"court"
/// Returns the tier score, or 0 for no match.
fn surface_match(token: &str, surface: &str) -> u8 {
    if token == surface {
        return TIER_EXACT;
    }
    let mut tv = morphological_variants(token);
    tv.push(token.to_string());
    let mut sv = morphological_variants(surface);
    sv.push(surface.to_string());
    if tv.iter().any(|t| sv.contains(t)) {
        return TIER_MORPH;
    }
    if token.len() >= 5 && surface.len() >= 5 && edit_distance(token, surface) <= 1 {
        return TIER_FUZZY;
    }
    0
}

/// Resolve a natural-language phrase to a component. Emergent: every seed surface
/// is expanded by morphology + tight fuzzy at match time, so inflections and typos
/// resolve without enumerating them. Best (component, tier) wins; ties keep
/// registry order.
pub fn resolve_component(text: &str) -> Option<&'static ComponentSpec> {
    let lower = text.to_lowercase();
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    let mut best: Option<(&'static ComponentSpec, u8)> = None;
    for comp in registry() {
        let mut score = 0u8;
        for tok in &tokens {
            for surf in &comp.surfaces {
                score = score.max(surface_match(tok, surf));
            }
        }
        if score > 0 && best.map(|(_, b)| score > b).unwrap_or(true) {
            best = Some((comp, score));
        }
    }
    best.map(|(c, _)| c)
}

/// Outcome of building a component: which leaves verified, whether it emits a
/// struct, plus the write + compile-gate result for the assembled module(s).
pub struct ComponentBuild {
    pub name: String,
    pub leaves_verified: Vec<String>,
    pub leaves_total: usize,
    pub has_struct: bool,
    pub outcome: WriteOutcome,
    /// Behavioral rung: `NotRun` when the component declares no smoke contract,
    /// else the `cargo test` result for its asserted runtime behavior.
    pub behavior: BehaviorStatus,
}

/// Result of the behavioral (`cargo test`) rung for a component.
#[derive(Debug)]
pub enum BehaviorStatus {
    /// The component declared no behavioral contract (bundle, or glue w/o smoke).
    NotRun,
    /// Smoke test ran and passed.
    Passed,
    /// Smoke test ran and failed (assertion or panic); carries the output.
    Failed(String),
    /// The gate could not run (infra error).
    Unverified(String),
}

impl BehaviorStatus {
    pub fn passed(&self) -> bool {
        matches!(self, BehaviorStatus::Passed)
    }
    /// True unless the smoke test actually ran and FAILED. `NotRun`/`Unverified`
    /// don't count as a behavioral failure.
    pub fn not_failed(&self) -> bool {
        !matches!(self, BehaviorStatus::Failed(_))
    }
    fn from_gate(status: CompileStatus) -> Self {
        match status {
            CompileStatus::Ok => BehaviorStatus::Passed,
            CompileStatus::Failed(e) => BehaviorStatus::Failed(e),
            CompileStatus::Unverified(e) => BehaviorStatus::Unverified(e),
        }
    }
}

impl ComponentBuild {
    /// True iff EVERY leaf verified AND the assembled module(s) compile.
    pub fn fully_verified(&self) -> bool {
        self.leaves_verified.len() == self.leaves_total
            && matches!(self.outcome.compile, CompileStatus::Ok)
    }
    /// True iff this component emitted a struct (structural component).
    pub fn produces_structure(&self) -> bool {
        self.has_struct
    }
    /// True iff the behavioral smoke test PASSED (the strongest guarantee: the
    /// assembled struct's runtime output is correct, not merely well-typed).
    pub fn behaves(&self) -> bool {
        self.behavior.passed()
    }
}

/// Synthesize the verified `(name, code)` pairs for a set of leaves via the
/// TRUSTED op path, de-duplicated by name (sibling components may share a leaf).
/// A leaf that fails to synthesize is DROPPED, never fabricated. Also returns the
/// verified leaf names in encounter order.
fn synth_leaves(
    bridge: &LinguigenesisBridge,
    leaf_sets: &[&[String]],
) -> (Vec<(String, String)>, Vec<String>) {
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut components: Vec<(String, String)> = Vec::new();
    let mut verified: Vec<String> = Vec::new();
    for leaves in leaf_sets {
        for leaf in *leaves {
            if !seen.insert(leaf.clone()) {
                continue;
            }
            if let Some(r) = bridge.synthesize_op_by_name(leaf) {
                if r.success {
                    verified.push(leaf.clone());
                    components.push((leaf.clone(), r.code));
                }
            }
        }
    }
    (components, verified)
}

/// Write a glue module verbatim and wire it into the crate's lib.rs. Idempotent on
/// the module name (a repeated glue module is skipped). Returns the written rel path
/// if it was newly wired.
fn write_and_wire_glue(root: &Path, glue: &GlueSpec) -> Result<Option<String>, String> {
    let glue_rel = format!("src/{}.rs", glue.module);
    let lib_path = root.join("src").join("lib.rs");
    let mut lib = std::fs::read_to_string(&lib_path).map_err(|e| e.to_string())?;
    let decl = format!("mod {};", glue.module);
    if lib.contains(&decl) {
        return Ok(None); // already wired
    }
    // Write the struct glue plus its behavioral contract (if any) in one file:
    // `cargo check` ignores the `#[cfg(test)]` module, `cargo test` runs it.
    let mut body = glue.code.clone();
    if let Some(smoke) = &glue.smoke {
        body.push('\n');
        body.push_str(smoke);
    }
    std::fs::write(root.join(&glue_rel), &body).map_err(|e| e.to_string())?;
    lib.push_str(&format!("\nmod {m};\npub use {m}::*;\n", m = glue.module));
    std::fs::write(&lib_path, &lib).map_err(|e| e.to_string())?;
    Ok(Some(glue_rel))
}

/// Build ONE component: synthesize its leaves, compose them into a module, and —
/// for a structural component — also emit the raw-Rust struct glue and wire it in.
/// The WHOLE crate is compiled (`cargo check`); a struct referencing a leaf that
/// failed, or mis-typed glue, fails compilation and is caught. Returns `Err` only
/// on write/infra failure or when nothing verified.
pub fn build_component(
    bridge: &LinguigenesisBridge,
    spec: &ComponentSpec,
    root: &Path,
) -> Result<ComponentBuild, String> {
    let (components, leaves_verified) = synth_leaves(bridge, &[spec.leaves.as_slice()]);
    if components.is_empty() {
        return Err(format!("component '{}': no leaf verified", spec.name));
    }
    let mut outcome = write_synthesized_project(root, &spec.name, &components)?;

    // Structural glue: only when the leaves themselves compiled (a struct over a
    // broken leaf would just fail again). Re-gate the WHOLE crate after wiring,
    // then — if the component declares a behavioral contract — run it (`cargo
    // test`), the rung above compilation.
    let mut behavior = BehaviorStatus::NotRun;
    if let Some(glue) = &spec.glue {
        if outcome.compile.is_ok() {
            if let Some(rel) = write_and_wire_glue(root, glue)? {
                outcome.written.push(rel);
                outcome.compile = compile_gate(root);
                if glue.smoke.is_some() && outcome.compile.is_ok() {
                    behavior = BehaviorStatus::from_gate(behavior_gate(root));
                }
            }
        }
    }

    Ok(ComponentBuild {
        name: spec.name.clone(),
        leaves_verified,
        leaves_total: spec.leaves.len(),
        has_struct: spec.glue.is_some(),
        outcome,
        behavior,
    })
}

/// Resolve ALL components a phrase mentions (each with a positive emergent match),
/// in registry order — the multi-component front door. "a counter and array
/// statistics" -> [counter, array_stats].
pub fn resolve_components(text: &str) -> Vec<&'static ComponentSpec> {
    let lower = text.to_lowercase();
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    registry()
        .iter()
        .filter(|comp| {
            tokens
                .iter()
                .any(|tok| comp.surfaces.iter().any(|s| surface_match(tok, s) > 0))
        })
        .collect()
}

/// Router-intent cue: does the phrase ASK to build a thing (vs perform an op)?
/// Kept minimal + deliberately about ROUTING, not NL resolution (which stays
/// emergent). A short article-led phrase ("a counter") also counts as a request to
/// construct that noun.
fn has_construction_cue(tokens: &[&str]) -> bool {
    const CUES: &[&str] = &[
        "build", "create", "make", "implement", "generate", "construct", "want",
        "need", "component", "struct", "module", "give",
    ];
    tokens.iter().any(|t| CUES.contains(t))
        || (tokens.len() <= 3 && matches!(tokens.first(), Some(&"a") | Some(&"an")))
}

/// ROUTER-SAFE component resolution for the auto-dispatcher. Stricter than
/// `resolve_components`: fires ONLY when the phrase carries a construction cue AND
/// the matching surface token resolves to NO coding op. The op filter is emergent
/// (asks the op resolver), so an ambiguous word like "count" — which resolves to
/// `array_sum` — never triggers a Counter build, while the distinctive noun
/// "counter" (resolves to no op) does. This is what makes it safe to hang off the
/// main router without hijacking operation requests.
pub fn route_component_build(
    bridge: &LinguigenesisBridge,
    query: &str,
) -> Vec<&'static ComponentSpec> {
    let lower = query.to_lowercase();
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    if !has_construction_cue(&tokens) {
        return Vec::new();
    }
    registry()
        .iter()
        .filter(|comp| {
            tokens.iter().any(|tok| {
                comp.surfaces.iter().any(|s| surface_match(tok, s) > 0)
                    && bridge.probe_resolution(tok).is_none()
            })
        })
        .collect()
}

/// Outcome of a multi-component project build.
pub struct ProjectBuild {
    pub components: Vec<String>,
    pub leaves_verified: Vec<String>,
    /// Glue module names emitted (structural components in the project).
    pub structs: Vec<String>,
    pub outcome: WriteOutcome,
    /// Behavioral rung for the whole crate: runs every structural component's
    /// smoke contract in one `cargo test`. `NotRun` when no component declares one.
    pub behavior: BehaviorStatus,
}

impl ProjectBuild {
    pub fn compiles(&self) -> bool {
        matches!(self.outcome.compile, CompileStatus::Ok)
    }
    pub fn behaves(&self) -> bool {
        self.behavior.passed()
    }
}

/// Build a MULTI-component project into ONE crate: the union of all components'
/// verified leaves plus each structural component's struct glue, wired into a
/// single lib.rs and compile-gated together. This is the planner's first symbolic
/// form — a prompt naming several concepts becomes one verified crate. Leaves are
/// synthesized once even when shared; glue modules are de-duplicated. Returns `Err`
/// only on write/infra failure or when no leaf across any component verified.
pub fn build_project(
    bridge: &LinguigenesisBridge,
    specs: &[&ComponentSpec],
    root: &Path,
) -> Result<ProjectBuild, String> {
    if specs.is_empty() {
        return Err("build_project: no components".to_string());
    }
    let leaf_sets: Vec<&[String]> = specs.iter().map(|s| s.leaves.as_slice()).collect();
    let (components, leaves_verified) = synth_leaves(bridge, &leaf_sets);
    if components.is_empty() {
        return Err("build_project: no leaf verified across any component".to_string());
    }
    let pkg = specs
        .iter()
        .map(|s| s.name.as_str())
        .collect::<Vec<_>>()
        .join("_");
    let mut outcome = write_synthesized_project(root, &pkg, &components)?;

    let mut structs: Vec<String> = Vec::new();
    let mut any_smoke = false;
    if outcome.compile.is_ok() {
        let mut wired_any = false;
        for spec in specs {
            if let Some(glue) = &spec.glue {
                if write_and_wire_glue(root, glue)?.is_some() {
                    outcome.written.push(format!("src/{}.rs", glue.module));
                    structs.push(glue.module.clone());
                    wired_any = true;
                    any_smoke |= glue.smoke.is_some();
                }
            }
        }
        if wired_any {
            outcome.compile = compile_gate(root);
        }
    }
    // Behavioral rung: one `cargo test` runs every structural smoke contract.
    let behavior = if any_smoke && outcome.compile.is_ok() {
        BehaviorStatus::from_gate(behavior_gate(root))
    } else {
        BehaviorStatus::NotRun
    };

    Ok(ProjectBuild {
        components: specs.iter().map(|s| s.name.clone()).collect(),
        leaves_verified,
        structs,
        outcome,
        behavior,
    })
}

/// Verdict from verifying an UNTRUSTED component proposal (e.g. one emitted by an
/// LLM planner). It is disposed of by the SAME compile + property gates a seed
/// component faces — this is the RLVR verifier that makes an untrusted proposer
/// safe: a hallucinated leaf, mistyped glue, a missing contract, or a lying
/// behavioral claim is REJECTED with a reason, never shipped.
#[derive(Debug)]
pub enum ProposalVerdict {
    Accepted {
        name: String,
        has_struct: bool,
        leaves: Vec<String>,
    },
    RejectedParse(String),
    RejectedNoLeaf(String),
    RejectedNoContract(String),
    RejectedCompile(String),
    RejectedBehavior(String),
}

impl ProposalVerdict {
    pub fn accepted(&self) -> bool {
        matches!(self, ProposalVerdict::Accepted { .. })
    }
}

/// Verify ONE untrusted component proposal (JSON for a single component) end to
/// end. Accepts ONLY if it parses, at least one leaf synthesizes, the crate
/// compiles, AND — for a structural proposal — it carries a behavioral contract
/// that PASSES. A structural proposal with no contract is rejected outright:
/// untrusted code doesn't get to skip proving itself. Untrusted in,
/// verified-or-rejected out — the safety property the whole LLM planner rests on.
pub fn verify_component_proposal(
    bridge: &LinguigenesisBridge,
    proposal_json: &str,
    root: &Path,
) -> ProposalVerdict {
    let specs = match parse_components_json(proposal_json) {
        Ok(s) => s,
        Err(e) => return ProposalVerdict::RejectedParse(e),
    };
    let spec = match specs.first() {
        Some(s) => s,
        None => return ProposalVerdict::RejectedParse("empty proposal".to_string()),
    };
    if let Some(glue) = &spec.glue {
        if glue.smoke.is_none() {
            return ProposalVerdict::RejectedNoContract(format!(
                "structural component '{}' ships no behavioral contract",
                spec.name
            ));
        }
    }
    let build = match build_component(bridge, spec, root) {
        Ok(b) => b,
        Err(e) => return ProposalVerdict::RejectedNoLeaf(e),
    };
    match &build.outcome.compile {
        CompileStatus::Failed(e) => return ProposalVerdict::RejectedCompile(e.clone()),
        CompileStatus::Unverified(e) => return ProposalVerdict::RejectedCompile(e.clone()),
        CompileStatus::Ok => {}
    }
    if spec.glue.is_some() && !build.behaves() {
        return ProposalVerdict::RejectedBehavior(format!("{:?}", build.behavior));
    }
    ProposalVerdict::Accepted {
        name: build.name,
        has_struct: build.has_struct,
        leaves: build.leaves_verified,
    }
}

/// The verified leaf ops the LLM proposer may use — each synthesizes with >=2
/// examples (confirmed by the leaf probe). Keeps the model from naming ops that
/// won't synthesize; anything off-menu is rejected by the verifier anyway.
pub fn proposable_leaves() -> Vec<String> {
    [
        "increment", "decrement", "add", "subtract", "multiply", "double", "triple",
        "negate", "square", "array_sum", "array_max", "array_min", "length",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// The RLVR loop CLOSED: ask the untrusted local model to PROPOSE a component for
/// `request` (using only verified leaves), then dispose of the proposal through the
/// same compile + behavior gates a seed component faces. Returns the verdict, or
/// `None` when the model lane is disabled/unreachable (nothing to verify). The
/// model can be arbitrarily unreliable — only an Accepted proposal is real.
pub fn propose_and_verify(
    bridge: &LinguigenesisBridge,
    request: &str,
    root: &Path,
) -> Option<ProposalVerdict> {
    let leaves = proposable_leaves();
    let json = crate::local_llm::propose_component(request, &leaves)?;
    Some(verify_component_proposal(bridge, &json, root))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("nsynth_component_{}_{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    #[test]
    fn resolves_components_from_prose() {
        // exact seed surfaces
        assert_eq!(
            resolve_component("give me some array statistics").unwrap().name,
            "array_stats"
        );
        assert_eq!(resolve_component("build a counter").unwrap().name, "counter");
        // EMERGENT — morphology reaches inflections with no per-form entry:
        // "counting"->count, "tallying"->tally, "counters"->counter.
        assert_eq!(resolve_component("counting the events").unwrap().name, "counter");
        assert_eq!(resolve_component("a tallying widget").unwrap().name, "counter");
        assert_eq!(resolve_component("wire up two counters").unwrap().name, "counter");
        // EMERGENT — tight fuzzy tolerates a one-char typo on a long word.
        assert_eq!(
            resolve_component("some statistcs please").unwrap().name,
            "array_stats"
        );
        // negatives — unrelated prose resolves to nothing (fuzzy stays tight).
        assert!(resolve_component("reverse an array").is_none());
        assert!(resolve_component("sort a list of names").is_none());
    }

    #[test]
    fn array_stats_bundle_synthesizes_and_compiles() {
        let bridge = LinguigenesisBridge::new();
        let spec = resolve_component("an array statistics module").expect("resolve stats");
        let root = temp_root("stats");
        let build = build_component(&bridge, spec, &root).expect("build");
        assert!(
            build.leaves_verified.len() >= 4,
            "expected >=4 verified leaves, got {:?}",
            build.leaves_verified
        );
        assert!(!build.produces_structure());
        assert!(
            build.outcome.compile.is_ok(),
            "component must compile: {:?}",
            build.outcome.compile
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn counter_structural_component_emits_a_struct_that_compiles() {
        let bridge = LinguigenesisBridge::new();
        let spec = resolve_component("a counter").expect("resolve counter");
        assert!(spec.glue.is_some(), "counter is structural");
        let root = temp_root("counter");
        let build = build_component(&bridge, spec, &root).expect("build");
        assert!(
            build.leaves_verified.contains(&"increment".to_string()),
            "increment leaf verified"
        );
        assert!(build.produces_structure(), "counter emits a struct");
        // The struct glue + the verified increment leaf compile TOGETHER.
        assert!(
            build.outcome.compile.is_ok(),
            "structural component must compile: {:?}",
            build.outcome.compile
        );
        // BEHAVIORAL RUNG: the smoke test ran and PASSED — three ticks reached 3,
        // proving the synthesized `increment` actually adds 1 (runtime, not types).
        assert!(
            build.behaves(),
            "counter must pass its behavioral contract: {:?}",
            build.behavior
        );
        // The struct is genuinely emitted, not stubbed.
        let glue = std::fs::read_to_string(root.join("src/counter.rs")).unwrap();
        assert!(glue.contains("pub struct Counter"), "struct present: {glue}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn accumulator_structural_component_folds_via_a_binary_leaf() {
        // Second structural component: a method that takes an ARGUMENT, backed by
        // the 2-arg `add` leaf — proves the glue pattern generalizes past Counter.
        let bridge = LinguigenesisBridge::new();
        let spec = resolve_component("an accumulator").expect("resolve accumulator");
        let root = temp_root("accum");
        let build = build_component(&bridge, spec, &root).expect("build");
        assert!(build.leaves_verified.contains(&"add".to_string()), "add leaf verified");
        assert!(build.produces_structure());
        assert!(build.outcome.compile.is_ok(), "compiles: {:?}", build.outcome.compile);
        // 5 + 3 + 10 == 18 at runtime -> the synthesized `add` genuinely sums.
        assert!(build.behaves(), "accumulator behavioral contract: {:?}", build.behavior);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn scaler_structural_component_configures_state_and_multiplies() {
        // Third structural shape: constructor takes an argument; state is a
        // configured operand applied via the 2-arg `multiply` leaf.
        let bridge = LinguigenesisBridge::new();
        let spec = resolve_component("a scaler").expect("resolve scaler");
        let root = temp_root("scaler");
        let build = build_component(&bridge, spec, &root).expect("build");
        assert!(build.leaves_verified.contains(&"multiply".to_string()), "multiply verified");
        assert!(build.produces_structure());
        assert!(build.outcome.compile.is_ok(), "compiles: {:?}", build.outcome.compile);
        // factor 3: 5 -> 15, proving the synthesized multiply behaves.
        assert!(build.behaves(), "scaler behavioral contract: {:?}", build.behavior);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn aggregator_collection_state_component_delegates_to_verified_reducers() {
        // Fourth structural shape: state is a COLLECTION (Vec), queries delegate to
        // the verified array reducers, and mean composes two verified leaves.
        let bridge = LinguigenesisBridge::new();
        let spec = resolve_component("an aggregator").expect("resolve aggregator");
        let root = temp_root("aggregator");
        let build = build_component(&bridge, spec, &root).expect("build");
        for leaf in ["array_sum", "array_max", "array_min", "length"] {
            assert!(
                build.leaves_verified.contains(&leaf.to_string()),
                "{leaf} verified: {:?}",
                build.leaves_verified
            );
        }
        assert!(build.produces_structure());
        assert!(build.outcome.compile.is_ok(), "compiles: {:?}", build.outcome.compile);
        // sum=20,max=8,min=2,count=4,mean=5 over [2,4,6,8] at runtime.
        assert!(build.behaves(), "aggregator behavioral contract: {:?}", build.behavior);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn stack_and_queue_collection_components_compile_and_behave() {
        let bridge = LinguigenesisBridge::new();
        for (phrase, tag, module) in [
            ("a stack", "stack", "stack"),
            ("a queue", "queue", "queue"),
        ] {
            let spec = resolve_component(phrase).unwrap_or_else(|| panic!("resolve {phrase}"));
            let root = temp_root(tag);
            let build = build_component(&bridge, spec, &root).expect("build");
            assert!(build.leaves_verified.contains(&"length".to_string()), "length verified");
            assert!(build.produces_structure());
            assert!(build.outcome.compile.is_ok(), "{tag} compiles: {:?}", build.outcome.compile);
            // LIFO/FIFO order + verified size proven at runtime.
            assert!(build.behaves(), "{tag} behavioral contract: {:?}", build.behavior);
            let glue = std::fs::read_to_string(root.join(format!("src/{module}.rs"))).unwrap();
            assert!(glue.contains("pub struct"), "{tag} struct present");
            let _ = std::fs::remove_dir_all(&root);
        }
    }

    #[test]
    fn multi_component_project_wires_struct_and_bundle_into_one_crate() {
        // One prompt naming two concepts resolves to two components...
        let specs = resolve_components("a counter and some array statistics");
        let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        assert!(
            names.contains(&"counter") && names.contains(&"array_stats"),
            "resolved both components: {names:?}"
        );
        // ...and builds into ONE verified crate.
        let bridge = LinguigenesisBridge::new();
        let root = temp_root("project");
        let build = build_project(&bridge, &specs, &root).expect("build");
        // union of leaves: increment (counter) + array reducers (stats)
        assert!(build.leaves_verified.contains(&"increment".to_string()), "{:?}", build.leaves_verified);
        assert!(
            build.leaves_verified.iter().any(|l| l == "array_sum"),
            "stats leaves present: {:?}",
            build.leaves_verified
        );
        // the structural component contributed its struct
        assert!(
            build.structs.contains(&"counter".to_string()),
            "counter struct emitted: {:?}",
            build.structs
        );
        // one crate, compiles together
        assert!(build.compiles(), "project compiles: {:?}", build.outcome.compile);
        // and its structural component's behavioral contract runs + passes in-crate.
        assert!(build.behaves(), "project behavior: {:?}", build.behavior);
        // struct + a bundle leaf share the SAME lib.rs
        let lib = std::fs::read_to_string(root.join("src/lib.rs")).unwrap();
        assert!(
            lib.contains("mod counter;") && lib.contains("mod increment;"),
            "one lib wires both: {lib}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn registry_includes_seeds_and_parse_rejects_garbage() {
        // The live registry is at least the four seeds.
        let names: Vec<&str> = registry().iter().map(|c| c.name.as_str()).collect();
        for n in ["array_stats", "counter", "accumulator", "scaler"] {
            assert!(names.contains(&n), "seed {n} present: {names:?}");
        }
        // A malformed data doc never poisons the registry — it just fails to parse.
        assert!(parse_components_json("{ not json").is_err());
        assert!(parse_components_json("[]").unwrap().is_empty());
    }

    #[test]
    fn data_defined_structural_component_synthesizes_compiles_and_behaves() {
        // A component authored purely in DATA (JSON) — no Rust const — builds a
        // verified crate end to end: proves the registry is genuinely data-driven.
        let doc = serde_json::json!([{
            "name": "countdown",
            "surfaces": ["countdown", "countdowns"],
            "leaves": ["decrement"],
            "glue": {
                "module": "countdown",
                "code": "use crate::decrement::decrement;\n\npub struct Countdown { n: i64 }\nimpl Countdown {\n    pub fn new(start: i64) -> Self { Countdown { n: start } }\n    pub fn step(&mut self) { self.n = decrement(self.n); }\n    pub fn get(&self) -> i64 { self.n }\n}\n",
                "smoke": "\n#[cfg(test)]\nmod countdown_behaves {\n    use super::Countdown;\n    #[test]\n    fn steps_down_to_one() {\n        let mut c = Countdown::new(3);\n        c.step();\n        c.step();\n        assert_eq!(c.get(), 1);\n    }\n}\n"
            }
        }])
        .to_string();

        let specs = parse_components_json(&doc).expect("parse data component");
        assert_eq!(specs.len(), 1);
        let spec = &specs[0];
        assert_eq!(spec.name, "countdown");
        assert!(spec.glue.is_some(), "data component is structural");

        let bridge = LinguigenesisBridge::new();
        let root = temp_root("countdown_data");
        let build = build_component(&bridge, spec, &root).expect("build");
        assert!(
            build.leaves_verified.contains(&"decrement".to_string()),
            "decrement leaf verified: {:?}",
            build.leaves_verified
        );
        assert!(
            build.outcome.compile.is_ok(),
            "data component compiles: {:?}",
            build.outcome.compile
        );
        // 3 -> step -> step -> 1: the synthesized decrement genuinely subtracts 1.
        assert!(build.behaves(), "data component behaves: {:?}", build.behavior);
        let glue = std::fs::read_to_string(root.join("src/countdown.rs")).unwrap();
        assert!(glue.contains("pub struct Countdown"), "struct present: {glue}");
        let _ = std::fs::remove_dir_all(&root);
    }

    // ---- RLVR verifier: untrusted proposal in -> verified-or-rejected out ----
    // Every hallucination class an LLM planner could emit is disposed of by the
    // SAME compile + behavior gates. These are the safety proofs.

    fn countdown_proposal(leaves: serde_json::Value, code: &str, smoke: serde_json::Value) -> String {
        serde_json::json!([{
            "name": "countdown",
            "surfaces": ["countdown"],
            "leaves": leaves,
            "glue": { "module": "countdown", "code": code, "smoke": smoke }
        }])
        .to_string()
    }

    const GOOD_CODE: &str = "use crate::decrement::decrement;\n\npub struct Countdown { n: i64 }\nimpl Countdown {\n    pub fn new(start: i64) -> Self { Countdown { n: start } }\n    pub fn step(&mut self) { self.n = decrement(self.n); }\n    pub fn get(&self) -> i64 { self.n }\n}\n";

    fn smoke_expecting(v: i64) -> serde_json::Value {
        serde_json::Value::String(format!(
            "\n#[cfg(test)]\nmod cd {{\n    use super::Countdown;\n    #[test]\n    fn t() {{\n        let mut c = Countdown::new(3);\n        c.step();\n        c.step();\n        assert_eq!(c.get(), {v});\n    }}\n}}\n"
        ))
    }

    #[test]
    fn proposal_accepted_when_it_survives_every_gate() {
        let bridge = LinguigenesisBridge::new();
        let root = temp_root("prop_ok");
        let json = countdown_proposal(
            serde_json::json!(["decrement"]),
            GOOD_CODE,
            smoke_expecting(1), // 3 -> step -> step -> 1: true
        );
        let v = verify_component_proposal(&bridge, &json, &root);
        assert!(v.accepted(), "should accept a correct proposal: {v:?}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn proposal_rejected_when_it_lies_about_behavior() {
        // Correct glue, but the contract asserts a FALSE result (99). Compiles, but
        // the behavior gate catches the lie.
        let bridge = LinguigenesisBridge::new();
        let root = temp_root("prop_lie");
        let json = countdown_proposal(
            serde_json::json!(["decrement"]),
            GOOD_CODE,
            smoke_expecting(99),
        );
        let v = verify_component_proposal(&bridge, &json, &root);
        assert!(
            matches!(v, ProposalVerdict::RejectedBehavior(_)),
            "must reject a lying contract: {v:?}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn proposal_rejected_when_glue_calls_a_leaf_it_did_not_declare() {
        // Glue calls `increment`, but only `decrement` is declared/synthesized -> the
        // increment module is absent -> compile fails.
        let bridge = LinguigenesisBridge::new();
        let root = temp_root("prop_badglue");
        let bad_code = "use crate::increment::increment;\n\npub struct Countdown { n: i64 }\nimpl Countdown {\n    pub fn new(start: i64) -> Self { Countdown { n: start } }\n    pub fn step(&mut self) { self.n = increment(self.n); }\n    pub fn get(&self) -> i64 { self.n }\n}\n";
        let json = countdown_proposal(
            serde_json::json!(["decrement"]),
            bad_code,
            smoke_expecting(1),
        );
        let v = verify_component_proposal(&bridge, &json, &root);
        assert!(
            matches!(v, ProposalVerdict::RejectedCompile(_)),
            "must reject glue that references an undeclared leaf: {v:?}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn proposal_rejected_when_leaf_does_not_synthesize() {
        let bridge = LinguigenesisBridge::new();
        let root = temp_root("prop_noleaf");
        // Bundle (no glue) whose only leaf is not a real op.
        let json = serde_json::json!([{
            "name": "phantom",
            "surfaces": ["phantom"],
            "leaves": ["totally_not_an_op_xyz"]
        }])
        .to_string();
        let v = verify_component_proposal(&bridge, &json, &root);
        assert!(
            matches!(v, ProposalVerdict::RejectedNoLeaf(_)),
            "must reject an unsynthesizable leaf: {v:?}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn proposal_rejected_when_structural_ships_no_contract() {
        let bridge = LinguigenesisBridge::new();
        let root = temp_root("prop_nocontract");
        let json = serde_json::json!([{
            "name": "countdown",
            "surfaces": ["countdown"],
            "leaves": ["decrement"],
            "glue": { "module": "countdown", "code": GOOD_CODE }
        }])
        .to_string();
        let v = verify_component_proposal(&bridge, &json, &root);
        assert!(
            matches!(v, ProposalVerdict::RejectedNoContract(_)),
            "structural code must prove itself: {v:?}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn proposal_rejected_when_json_is_garbage() {
        let bridge = LinguigenesisBridge::new();
        let root = temp_root("prop_garbage");
        let v = verify_component_proposal(&bridge, "{ not json", &root);
        assert!(matches!(v, ProposalVerdict::RejectedParse(_)), "{v:?}");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn propose_and_verify_is_inert_without_a_model() {
        // With no model endpoint the RLVR loop yields None (nothing proposed), so
        // the CI path never depends on a running server.
        std::env::remove_var("NSYNTH_LOCAL_LLM_URL");
        let bridge = LinguigenesisBridge::new();
        let root = temp_root("propose_inert");
        assert!(propose_and_verify(&bridge, "a thing that squares numbers", &root).is_none());
        assert!(!proposable_leaves().is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }
}
