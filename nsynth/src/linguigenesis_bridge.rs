//! Linguigenesis integration for nSynth
//!
//! Pure Rust NL→Code synthesis using Linguigenesis comprehension.
//! No external APIs, no Python - zero hallucination by construction.

use crate::benchmark::{Example, Problem, Value};
use linguigenesis_core::{
    belief::BeliefState,
    coding_comprehension::{CodingComprehension, ComprehensionOutcome},
    coding_dialogue::{
        apply_clarification, build_clarifications, format_clarification_prompt,
        needs_clarification, ClarificationField, ClarificationQuestion,
    },
    coding_requirements::{
        CompositionPlan, FilterPred, LiteralValue, OpRef, OpRole, ProjectPlan,
        SynthesisRequirement,
    },
    comprehension::Comprehension,
    entity_resolution::EntityResolver,
    computing_knowledge_import::merge_computing_knowledge,
    reasoning::{AnalogyReasoner, KnowledgeQA},
    registry::Registry,
};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock, RwLockWriteGuard};
use std::time::SystemTime;

thread_local! {
    /// Set while inside the local-LLM fallback so the symbolic path it re-invokes
    /// (Mode A' rephrase → synthesize_from_description) cannot re-enter the LLM
    /// fallback — no infinite recursion.
    static IN_LLM_FALLBACK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Linguigenesis bridge for NL→Code synthesis
pub struct LinguigenesisBridge {
    /// Comprehension engine
    comprehension: Arc<RwLock<Comprehension>>,
    /// Knowledge QA engine
    qa: Arc<KnowledgeQA>,
    /// Analogy reasoner
    analogy: Arc<AnalogyReasoner>,
    /// Code entity registry
    registry: Arc<RwLock<Registry>>,
    /// Registry file path for auto-update
    registry_path: Option<PathBuf>,
    /// Last modification time
    last_modified: Option<SystemTime>,
}

/// A component whose behavior was strict-verified against LLM-proposed examples.
/// `examples` is the FULL proposed set (seed + holdouts) — intended for emitting
/// reproduction tests (PIECE 3/4), NOT a training subset.
pub struct VerifiedComponent {
    pub name: String,
    pub result: crate::solver::SolveResult,
    pub examples: Vec<crate::benchmark::Example>,
}

/// Map one raw LLM-proposed JSON value to a runtime `benchmark::Value`. Byte-for-byte
/// the same coercion as the inline closure in `synthesize_via_llm_examples`
/// (bool → Bool, i64 → Int, all-int array → int_array, else str → Str); any value
/// that fits none of those — a float, a mixed array, a nested object — yields `None`
/// so the caller skips that example rather than coercing it wrongly.
fn json_to_bench_value(v: &serde_json::Value) -> Option<crate::benchmark::Value> {
    use crate::benchmark::Value;
    if let Some(b) = v.as_bool() {
        return Some(Value::Bool(b));
    }
    if let Some(i) = v.as_i64() {
        return Some(Value::Int(i));
    }
    if let Some(arr) = v.as_array() {
        let ints: Option<Vec<i64>> = arr.iter().map(|x| x.as_i64()).collect();
        return Some(Value::int_array(&ints?));
    }
    v.as_str().map(|s| Value::Str(s.to_string()))
}

impl LinguigenesisBridge {
    /// Resolve a Linguigenesis data file by name in a LOCATION-INDEPENDENT way.
    ///
    /// The previous path-finders (`find_*_path`) resolved only against the
    /// process CWD and `$HOME`, so launching the agent from any directory
    /// outside `nsynth/` (or under a non-default `HOME`) failed to locate the
    /// registry and the agent silently degraded to "everything is unknown".
    ///
    /// We now probe, in order:
    ///   1. A COMPILE-TIME ABSOLUTE base derived from this crate's
    ///      `CARGO_MANIFEST_DIR`. nSynth lives at `<root>/nsynth`, the data at
    ///      `<root>/../linguigenesis/data` relative to it, i.e.
    ///      `<MANIFEST_DIR>/../../linguigenesis/data/<file>`. This is fixed at
    ///      build time and does not depend on CWD/HOME at all.
    ///   2. The directory of the running executable (`current_exe`), walking up
    ///      to find a sibling `linguigenesis/data` — covers relocated binaries
    ///      whose source tree moved but kept the `linguigenesis` sibling.
    ///   3. The legacy CWD-relative and `$HOME`-relative paths as fallback (so
    ///      nothing that worked before stops working).
    ///
    /// Returns the first existing path, or `None` if the file is nowhere.
    fn locate_data_file(file_name: &str) -> Option<PathBuf> {
        // (1) compile-time absolute base — cwd/HOME-independent.
        let manifest_base = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../linguigenesis/data")
            .join(file_name);
        if manifest_base.exists() {
            return Some(manifest_base);
        }

        // (2) executable-relative: walk up from the binary looking for a
        //     sibling `linguigenesis/data/<file>`.
        if let Ok(exe) = std::env::current_exe() {
            let mut dir = exe.parent();
            while let Some(d) = dir {
                let candidate = d.join("linguigenesis/data").join(file_name);
                if candidate.exists() {
                    return Some(candidate);
                }
                dir = d.parent();
            }
        }

        // (3) legacy fallbacks: CWD-relative then $HOME-relative.
        let relative = PathBuf::from("../../linguigenesis/data").join(file_name);
        if relative.exists() {
            return Some(relative);
        }
        if let Ok(home) = std::env::var("HOME") {
            let home_path = PathBuf::from(home)
                .join("projects/linguigenesis/data")
                .join(file_name);
            if home_path.exists() {
                return Some(home_path);
            }
        }
        let current = PathBuf::from("linguigenesis/data").join(file_name);
        if current.exists() {
            return Some(current);
        }
        None
    }

    /// Create new bridge with auto-loading from Linguigenesis registry
    pub fn new() -> Self {
        // Try to load from Linguigenesis data directory
        let linguigenesis_path = Self::find_registry_path();

        let (registry, modified) = if let Some(path) = &linguigenesis_path {
            Self::load_registry_with_fallback(path)
        } else {
            Self::load_registry_with_fallback(Path::new(""))
        };

        let comprehension = Arc::new(RwLock::new(Comprehension::new(registry.clone())));
        let qa = Arc::new(KnowledgeQA::new(registry.clone()));
        let analogy = Arc::new(AnalogyReasoner::new(registry.clone()));

        Self {
            comprehension,
            qa,
            analogy,
            registry: Arc::new(RwLock::new(registry)),
            registry_path: linguigenesis_path,
            last_modified: modified,
        }
    }

    /// Find Linguigenesis registry path (location-independent; see
    /// [`locate_data_file`]).
    fn find_registry_path() -> Option<PathBuf> {
        // Opt-in override: point the base registry at any entities-format file
        // (e.g. a large word graph) via NSYNTH_BASE_REGISTRY — an absolute/relative
        // path or a bare filename resolved under the data dir. Default behaviour
        // (look for `registry.json`) is unchanged when unset, so the large 108MB
        // dump is never loaded unless explicitly requested.
        if let Ok(spec) = std::env::var("NSYNTH_BASE_REGISTRY") {
            let spec = spec.trim();
            if !spec.is_empty() {
                let direct = PathBuf::from(spec);
                if direct.exists() {
                    return Some(direct);
                }
                if let Some(found) = Self::locate_data_file(spec) {
                    return Some(found);
                }
            }
        }
        Self::locate_data_file("registry.json")
    }

    /// TEST-ONLY constructor: identical to [`new`] but WITHOUT the WordNet
    /// coding-edges file merged. Used by the recall-lift benchmark to compute the
    /// registry-only baseline so the lift is attributable solely to the edges.
    /// (Not env-driven, to stay race-free under parallel `cargo test`.)
    pub fn new_without_wordnet_edges() -> Self {
        let linguigenesis_path = Self::find_registry_path();
        let (registry, modified) = if let Some(path) = &linguigenesis_path {
            Self::load_registry_with_fallback_opt(path, false)
        } else {
            Self::load_registry_with_fallback_opt(Path::new(""), false)
        };
        let comprehension = Arc::new(RwLock::new(Comprehension::new(registry.clone())));
        let qa = Arc::new(KnowledgeQA::new(registry.clone()));
        let analogy = Arc::new(AnalogyReasoner::new(registry.clone()));
        Self {
            comprehension,
            qa,
            analogy,
            registry: Arc::new(RwLock::new(registry)),
            registry_path: linguigenesis_path,
            last_modified: modified,
        }
    }

    /// Load registry with fallback to code entities if file not found
    fn load_registry_with_fallback(path: &Path) -> (Registry, Option<SystemTime>) {
        Self::load_registry_with_fallback_opt(path, true)
    }

    fn load_registry_with_fallback_opt(
        path: &Path,
        include_wordnet: bool,
    ) -> (Registry, Option<SystemTime>) {
        if !path.as_os_str().is_empty() && path.exists() {
            match Registry::from_json_auto(path) {
                Ok((mut registry, modified)) => {
                    eprintln!(
                        "[Linguigenesis] Loaded {} entities from {}",
                        registry.stats().total_entities,
                        path.display()
                    );
                    Self::merge_coding_registry_opt(&mut registry, include_wordnet);
                    return (registry, modified);
                }
                Err(e) => {
                    eprintln!(
                        "[Linguigenesis] Failed to load registry: {}, using fallback",
                        e
                    );
                }
            }
        }

        let mut registry = Registry::new();
        Self::merge_coding_registry_opt(&mut registry, include_wordnet);
        if registry.stats().total_entities == 0 {
            eprintln!(
                "[Linguigenesis] No coding registry loaded — set ../../linguigenesis/data/coding_registry.json"
            );
        }
        (registry, None)
    }

    fn find_coding_registry_path() -> Option<PathBuf> {
        Self::locate_data_file("coding_registry.json")
    }

    fn find_computing_knowledge_path() -> Option<PathBuf> {
        Self::locate_data_file("computing_knowledge.json")
    }
    fn find_mined_capabilities_path() -> Option<PathBuf> {
        Self::locate_data_file("mined_capabilities.json")
    }

    /// Public path lookup for runtime introspection (MCP / CLI capabilities).
    pub fn data_file_path(file_name: &str) -> Option<PathBuf> {
        Self::locate_data_file(file_name)
    }

    /// Find the WordNet coding-edges file (additive synonym/similar closure for
    /// existing ops). Resolved location-independently via [`locate_data_file`].
    fn find_wordnet_edges_path() -> Option<PathBuf> {
        Self::locate_data_file("wordnet_coding_edges.json")
    }

    fn merge_coding_registry(registry: &mut Registry) {
        Self::merge_coding_registry_opt(registry, true);
    }

    /// Additively declare the value-TYPE vocabulary the fail-closed gate compares
    /// operand types against. The registry ships operation entities and an "array"
    /// noun, but NO `Type` entities, so the gate's generic type-mismatch check
    /// (`mentioned_value_type`, which keys on `EntityType::Type`) is dormant and a
    /// request like "reverse a STRING" silently resolves to the i64-array `reverse`
    /// — a confidently-wrong domain answer the prior code documented as an
    /// acceptable over-acceptance pending exactly this Type entity
    /// (see `nl_reverse_a_string_is_a_documented_type_case`).
    ///
    /// This declares the `string` type (with its signature surface forms via
    /// `signature_aliases`) ONCE, ONLY if absent — additive, idempotent, and on
    /// the registry path. It is a TYPE VOCABULARY entry, NOT a phrase→refuse map:
    /// the gate still decides emergently by comparing the resolved op's signature
    /// against the mentioned type's surface forms. A genuine string-typed op (none
    /// ship today) whose signature carried "string" would satisfy the check and be
    /// accepted; only a type/domain mismatch refuses.
    fn ensure_value_type_vocabulary(registry: &mut Registry) {
        use linguigenesis_core::entity::{Entity, EntityType};
        // (lemma, definition, signature_aliases) for each value-TYPE the fail-closed
        // gate compares a resolved op's signature against. NL-BRIDGE-1 adds `float`
        // alongside `string` so a genuine float/string type-mismatch still refuses
        // (the gate keys on `EntityType::Type`); the seed float/string ops carry the
        // matching surface forms so a CORRECT-domain request is accepted, only a
        // mismatch refuses. Additive + idempotent: each is declared once, only if
        // its lemma is absent.
        let value_types = [
            (
                "string",
                "Sequence of characters (text value)",
                "string,str,&str",
            ),
            (
                "float",
                "Real (floating-point) number value",
                "f64,f32,float,double",
            ),
        ];
        let mut next_id = registry.stats().total_entities as u64 + 5000;
        for (lemma, def, aliases) in value_types {
            if registry.get_by_lemma(lemma).is_some() {
                continue;
            }
            let mut entity = Entity::new(next_id, lemma.to_string(), EntityType::Type);
            next_id += 1;
            entity.add_definition(def.to_string());
            // Signature surface forms the gate matches against an op's signature.
            entity.add_property("signature_aliases".to_string(), aliases.to_string());
            if let Err(e) = registry.add_entity(entity) {
                eprintln!("[Linguigenesis] {lemma} Type vocabulary add warning: {e}");
            }
        }
    }

    /// Collision-safe merge of the WordNet edge file into an ALREADY-populated
    /// registry. Each NEW closure word is re-added with a FRESH high id (base +
    /// 1000+, mirroring `merge_computing_knowledge`) so it never overwrites a
    /// like-id coding entity; its `synonym`/`similar` relations are then re-linked
    /// BY LEMMA against the real (example-bearing) seed entity in `registry`.
    /// Entities whose lemma already exists (the co-declared seed stubs, or any word
    /// the hand-table already owns) are left untouched — purely additive.
    fn merge_wordnet_edges_collision_safe(registry: &mut Registry, edges: &Registry) {
        use linguigenesis_core::entity::{Entity, RelationType};
        let mut next_id = registry.stats().total_entities as u64 + 1000;
        // (source_lemma, rel_type, target_lemma) links to establish after adding.
        let mut pending: Vec<(String, RelationType, String)> = Vec::new();

        for src in edges.all_entities() {
            // Skip anything already present (seed stubs, hand-table words): additive.
            if registry.get_by_lemma(&src.lemma).is_some() {
                continue;
            }
            // Re-create the entity with a fresh, non-colliding id. Carry over the
            // type, definitions, and properties (incl. wn_synset provenance) but
            // DROP the edge-file-local relation ids — they are re-linked by lemma
            // below so they bind to the real seed, never a stale numeric target.
            let mut entity = Entity::new(next_id, src.lemma.clone(), src.entity_type.clone());
            next_id += 1;
            for def in &src.definitions {
                entity.add_definition(def.clone());
            }
            for (k, v) in &src.properties {
                entity.add_property(k.clone(), v.clone());
            }
            if let Err(e) = registry.add_entity(entity) {
                eprintln!("[Linguigenesis] wordnet edge add warning ({}): {}", src.lemma, e);
                continue;
            }
            // Queue this word's relations to its seed target(s), resolved by lemma
            // within the EDGE registry (so the underscore-free seed lemma is known).
            for (rel_type, target_ids) in &src.relations {
                for tid in target_ids {
                    if let Some(target) = edges.get_entity(*tid) {
                        pending.push((src.lemma.clone(), rel_type.clone(), target.lemma.clone()));
                    }
                }
            }
        }

        for (source, rel_type, target) in pending {
            // Links only succeed when BOTH lemmas exist in `registry`; the target
            // is the real seed op (present from coding_registry), so the closure
            // word now resolves through a genuine synonym/similar edge to the seed.
            let _ = registry.link_lemma_relation(&source, rel_type, &target);
        }
    }

    /// Collision-safe merge of the AUTO-MINED capability registry into an
    /// ALREADY-populated registry. Each mined entity is re-added with a FRESH high
    /// id (base + 1000+), mirroring `merge_wordnet_edges_collision_safe`, so it
    /// never overwrites a like-id coding entity. Definitions + properties (incl.
    /// the machine-generated `example_cases`, `input_types`, `output_type`,
    /// `nsynth_category`, `default_fn_name`) are carried over so the mined op is a
    /// full, synthesizable capability; relations (synonym/domain) are re-linked BY
    /// LEMMA. If the mined lemma ALREADY exists (a hand seed not yet removed), its
    /// properties are overlaid only when the existing entity lacks `example_cases`
    /// — purely additive, never destructive.
    fn merge_mined_collision_safe(registry: &mut Registry, mined: &Registry) {
        use linguigenesis_core::entity::{Entity, RelationType};
        let mut next_id = registry.stats().total_entities as u64 + 1000;
        let mut pending: Vec<(String, RelationType, String)> = Vec::new();

        for src in mined.all_entities() {
            if let Some(existing) = registry.get_by_lemma(&src.lemma) {
                // Lemma already present (un-removed seed): overlay missing props.
                if existing.get_property("example_cases").is_none()
                    && src.get_property("example_cases").is_some()
                {
                    let _ = registry.overlay_entity_properties(&src.lemma, &src);
                }
                continue;
            }
            let mut entity = Entity::new(next_id, src.lemma.clone(), src.entity_type.clone());
            next_id += 1;
            for def in &src.definitions {
                entity.add_definition(def.clone());
            }
            for (k, v) in &src.properties {
                entity.add_property(k.clone(), v.clone());
            }
            if let Err(e) = registry.add_entity(entity) {
                eprintln!("[Linguigenesis] mined capability add warning ({}): {}", src.lemma, e);
                continue;
            }
            // Re-link this op's relations to their targets BY LEMMA (resolved in
            // the mined registry so synonym/domain lemmas are known).
            for (rel_type, target_ids) in &src.relations {
                for tid in target_ids {
                    if let Some(target) = mined.get_entity(*tid) {
                        pending.push((src.lemma.clone(), rel_type.clone(), target.lemma.clone()));
                    }
                }
            }
        }

        for (source, rel_type, target) in pending {
            let _ = registry.link_lemma_relation(&source, rel_type, &target);
        }
    }

    fn merge_coding_registry_opt(registry: &mut Registry, include_wordnet: bool) {
        if let Some(path) = Self::find_coding_registry_path() {
            if let Ok(coding) = Registry::from_json(&path) {
                if let Err(e) = registry.merge_registry(&coding) {
                    eprintln!("[Linguigenesis] coding_registry merge warning: {}", e);
                }
            }
        }
        // AUTO-MINED capabilities (NL-BRIDGE-2): the engine's own synthesizable
        // operator surface (string_synth `SExpr`, array_transform `ReorderKind`),
        // reflected offline into the same `coding_registry.json` schema with
        // auto-run example_cases, and loaded through the EXACT same proven path as
        // the hand registry above. Runs AFTER coding_registry so mined entities
        // augment/overlay it (e.g. supply the `lowercase`/`trim` ops the hand
        // registry never seeded). Self-growing: re-running the miner after adding
        // an operator variant grows the NL vocabulary with no code edit here.
        if let Some(path) = Self::find_mined_capabilities_path() {
            match Registry::from_json(&path) {
                Ok(mined) => {
                    // COLLISION-SAFE merge (NOT `merge_registry`): `from_json`
                    // numbers the mined file's entities 1..N, and `add_entity`
                    // inserts at the donor id, which OVERWRITES the like-id
                    // coding_registry entities in the id-keyed map (corrupting e.g.
                    // `add`). We re-add each mined entity with a FRESH high id —
                    // exactly as `merge_wordnet_edges_collision_safe` /
                    // `merge_computing_knowledge` do — carrying its definitions,
                    // properties (incl. the auto-mined `example_cases`) and
                    // relations re-linked by lemma.
                    Self::merge_mined_collision_safe(registry, &mined);
                }
                Err(e) => {
                    eprintln!("[Linguigenesis] mined_capabilities load warning: {}", e);
                }
            }
        }
        // 3rd merge: WordNet synonym/similar closure edges for EXISTING ops. Must
        // run AFTER coding_registry so each seed op already exists in the merged
        // registry — the closure word's synonym->seed edge then links to the real,
        // example-bearing seed entity.
        //
        // COLLISION-SAFE merge (NOT `merge_registry`): `Registry::from_json`
        // numbers the edge file's entities 1..N, and `merge_registry` PRESERVES
        // those low ids when re-adding, which OVERWRITES the like-id coding_registry
        // entities in the id-keyed map (corrupting e.g. `add`). We instead allocate
        // FRESH high ids — exactly as `merge_computing_knowledge` does (base +
        // 1000) — so the closure words never collide. Seed stubs in the edge file
        // already exist (skipped); their only purpose is to be the `synonym`
        // targets, which we re-link by lemma against the real seed.
        if include_wordnet {
            if let Some(path) = Self::find_wordnet_edges_path() {
                match Registry::from_json(&path) {
                    Ok(edges) => {
                        Self::merge_wordnet_edges_collision_safe(registry, &edges);
                    }
                    Err(e) => {
                        eprintln!("[Linguigenesis] wordnet_coding_edges load warning: {}", e);
                    }
                }
            }
        }
        if let Some(path) = Self::find_computing_knowledge_path() {
            if let Err(e) = merge_computing_knowledge(registry, &path) {
                eprintln!("[Linguigenesis] computing_knowledge merge warning: {}", e);
            }
        }
        // Activate the dormant emergent type-mismatch gate by declaring the value-
        // type vocabulary it compares against (additive, idempotent).
        Self::ensure_value_type_vocabulary(registry);
        // Repair cross-file synonym danglers: edges DECLARED in coding_registry.json
        // that point to ops living in mined_capabilities.json (e.g. flip/invert ->
        // reverse) are dropped at coding load (target absent then). Now that mined is
        // merged, restore the author-declared edges so those surfaces resolve. Only
        // already-declared edges are re-linked, so this cannot widen resolution.
        if let Some(path) = Self::find_coding_registry_path() {
            match registry.relink_declared_edges_from_json(&path) {
                Ok(n) if n > 0 => {
                    eprintln!("[Linguigenesis] relinked {} declared coding synonym edge(s)", n)
                }
                _ => {}
            }
        }
        if let Err(errors) =
            linguigenesis_core::coding_registry_validate::validate_coding_registry(registry)
        {
            eprintln!(
                "[Linguigenesis] coding registry validation warnings: {}",
                errors.join("; ")
            );
        }
    }

    /// Check for registry updates and reload if needed
    pub fn check_and_update(&mut self) -> Result<(), String> {
        let Some(path) = &self.registry_path else {
            return Ok(()); // No file to watch
        };

        let metadata =
            std::fs::metadata(path).map_err(|e| format!("Failed to read file metadata: {}", e))?;

        let modified = metadata
            .modified()
            .map_err(|e| format!("Failed to get modified time: {}", e))?;

        if let Some(last) = self.last_modified {
            if modified <= last {
                return Ok(()); // No update needed
            }
        }

        // Need to update
        eprintln!("[Linguigenesis] Registry updated, reloading...");
        let (new_registry, new_modified) = Self::load_registry_with_fallback(path);

        *self
            .registry
            .write()
            .map_err(|_| "Lock error".to_string())? = new_registry.clone();
        *self
            .comprehension
            .write()
            .map_err(|_| "Lock error".to_string())? = Comprehension::new(new_registry.clone());
        self.qa = Arc::new(KnowledgeQA::new(new_registry.clone()));
        self.analogy = Arc::new(AnalogyReasoner::new(new_registry));

        self.last_modified = new_modified;
        eprintln!("[Linguigenesis] Reload complete");

        Ok(())
    }

    /// EXPLICIT load-failure surface (FIX A): the coding registry is the source
    /// of every programming operation the agent can resolve. If it failed to
    /// load (file not found from ANY of the location-independent probes, or
    /// loaded but carries zero Function/Operator entities), the agent would
    /// otherwise silently degrade — every request resolves to nothing and is
    /// reported as "workflow unknown / clarification". Callers consult this to
    /// surface a real error instead of pretending every op is unknown.
    ///
    /// Returns `Some(message)` when no operations are available, `None` when the
    /// registry is healthy.
    pub fn registry_load_error(&self) -> Option<String> {
        use linguigenesis_core::entity::EntityType;
        let registry = self.registry.read().ok()?;
        let has_ops = !registry.get_by_type(&EntityType::Function).is_empty()
            || !registry.get_by_type(&EntityType::Operator).is_empty();
        if has_ops {
            None
        } else {
            Some(format!(
                "Linguigenesis coding registry failed to load (0 operations available): \
                 looked for coding_registry.json via compile-time base \
                 '{}/../../linguigenesis/data', the executable's sibling \
                 'linguigenesis/data', then CWD/$HOME fallbacks. The agent cannot \
                 resolve any operation until the registry is reachable.",
                env!("CARGO_MANIFEST_DIR")
            ))
        }
    }

    /// Create bridge with custom registry
    pub fn with_registry(registry: Registry) -> Self {
        let comprehension = Arc::new(RwLock::new(Comprehension::new(registry.clone())));
        let qa = Arc::new(KnowledgeQA::new(registry.clone()));
        let analogy = Arc::new(AnalogyReasoner::new(registry.clone()));

        Self {
            comprehension,
            qa,
            analogy,
            registry: Arc::new(RwLock::new(registry)),
            registry_path: None,
            last_modified: None,
        }
    }

    /// Clone the loaded KVRM registry (for clarification / comprehension helpers).
    pub fn registry_clone(&self) -> Result<Registry, BridgeError> {
        Ok(self
            .registry
            .read()
            .map_err(|_| BridgeError::LockError)?
            .clone())
    }

    /// EMERGENT fail-closed gate (FIX C), exposed for callers (the agent's
    /// `run_synthesis`) that obtained a `SynthesisRequirement` via
    /// `comprehend_outcome` and therefore BYPASSED the gate baked into
    /// `nl_to_requirement`. Runs the exact same
    /// [`unsound_confident_solve_categorized`] check (domain/type/operation-
    /// identity + the >=2-example floor, derived from the resolver + registry
    /// signatures — NOT a phrase blocklist).
    ///
    /// Returns `Some(reason)` ONLY for HARD refusals — a confidently-WRONG
    /// resolution (out-of-domain, type/signature mismatch, thin scalar spec, or no
    /// operation resolved). The SOFT `CompositionUnsupported` case (the request
    /// names a second op the array-pipeline path did not build, e.g. the
    /// conjunction "and" in "doubles AND squares a number") is NOT refused here:
    /// the single-op solver still synthesizes the primary op, matching the agent's
    /// long-standing single-function behaviour. (`nl_to_requirement` keeps refusing
    /// on every category via the back-compat `unsound_confident_solve`.)
    pub fn fail_closed_reason(&self, input: &str, req: &SynthesisRequirement) -> Option<String> {
        let registry = self.registry.read().ok()?;
        match unsound_confident_solve_categorized(input, req, &registry) {
            Some(r) if r.category == GateCategory::Hard => Some(r.reason),
            _ => None,
        }
    }

    /// Parse NL into registry-derived synthesis requirements (KVRM only).
    pub fn nl_to_requirement(&self, input: &str) -> Result<SynthesisRequirement, BridgeError> {
        let registry = self
            .registry
            .read()
            .map_err(|_| BridgeError::LockError)?
            .clone();
        let mut coding = CodingComprehension::new(registry);
        let mut req = coding.comprehend(input);
        // STATEFUL RE-TARGET (UNWALL-1-STATEFUL-NL): a request whose primary op is a
        // collection→scalar REDUCE (array→int example shape) but which ALSO carries
        // a STRUCTURAL per-tick-state operand signal (a token resolves to a
        // `nsynth_category=stateful` entity) is a per-tick stateful reducer, not a
        // plain array reduce. Re-target it to the mined stateful capability whose
        // engine reducer BEHAVIORALLY reproduces the resolved reduce op's outputs
        // (state=0 ⇒ f(0,arr)=g(arr)). Both signals are structural: the reduce shape
        // is read from example types, the stateful signal from a registry-resolved
        // operand class, and the op identity from a BEHAVIORAL match — never a
        // phrase→op table. Plain reduces (no stateful operand token) are untouched.
        if let Some(stateful) = retarget_stateful_reducer(input, &req, coding.registry()) {
            req = stateful;
        }
        if req.examples.is_empty() {
            let questions = build_clarifications(&req, coding.registry());
            if !questions.is_empty() {
                return Err(BridgeError::ClarificationNeeded {
                    partial: req,
                    questions,
                });
            }
            return Err(BridgeError::ParseError(if req.unresolved.is_empty() {
                "no examples derived from registry".to_string()
            } else {
                req.unresolved.join("; ")
            }));
        }
        if needs_clarification(&req) {
            let questions = build_clarifications(&req, coding.registry());
            if !questions.is_empty() {
                return Err(BridgeError::ClarificationNeeded {
                    partial: req,
                    questions,
                });
            }
        }
        // SOUNDNESS GATE (P0-NL fail-closed): a registry NL match whose only
        // evidence is the op's canned example(s) must NOT be reported as a
        // confident solve when STRUCTURAL signals show the request was not
        // actually understood. All signals are computed from `req` + the raw
        // request — no phrase blocklist, no request->refuse map.
        if let Some(reason) = unsound_confident_solve(input, &req, coding.registry()) {
            let mut partial = req;
            // Reuse the existing "no operation" clarification path so the
            // downgrade flows through build_clarifications unchanged.
            partial.unresolved.push(reason);
            let questions = build_clarifications(&partial, coding.registry());
            return Err(BridgeError::ClarificationNeeded { partial, questions });
        }
        Ok(req)
    }

    /// Comprehend NL with clarification outcome (ready or typed questions).
    pub fn comprehend_outcome(&self, input: &str) -> Result<ComprehensionOutcome, BridgeError> {
        let registry = self
            .registry
            .read()
            .map_err(|_| BridgeError::LockError)?
            .clone();
        let mut coding = CodingComprehension::new(registry);
        let outcome = coding.comprehend_outcome(input);
        // STATEFUL RE-TARGET (UNWALL-1-STATEFUL-NL): apply the SAME structural
        // re-target the `nl_to_requirement` front door uses, so the PRODUCT path
        // (`CodingAgentSession::handle_query` → `comprehend_outcome`) also routes a
        // per-tick stateful reducer to the real stateful synthesis instead of a
        // plain array reduce. A re-targeted request is unambiguously Ready (its
        // behaviorally-matched stateful examples are the spec), so it also resolves
        // an ambiguous reduce-shaped clarification.
        let req_ref = match &outcome {
            ComprehensionOutcome::Ready(req) => req,
            ComprehensionOutcome::NeedsClarification(req, _) => req,
        };
        if let Some(stateful) = retarget_stateful_reducer(input, req_ref, coding.registry()) {
            return Ok(ComprehensionOutcome::Ready(stateful));
        }
        Ok(outcome)
    }

    /// Apply a clarification answer and return an updated requirement.
    pub fn apply_clarification(
        &self,
        partial: &mut SynthesisRequirement,
        field: ClarificationField,
        answer: &str,
    ) -> Result<(), BridgeError> {
        let registry = self
            .registry
            .read()
            .map_err(|_| BridgeError::LockError)?
            .clone();
        if apply_clarification(partial, &field, answer, &registry) {
            Ok(())
        } else {
            Err(BridgeError::InvalidInput(format!(
                "could not apply clarification '{}' for {:?}",
                answer, field
            )))
        }
    }

    /// Parse NL and generate synthesis examples from registry `example_cases`.
    pub fn nl_to_examples(&self, input: &str) -> Result<Vec<Example>, BridgeError> {
        let req = self.nl_to_requirement(input)?;
        synthesis_requirement_to_examples(&req)
    }

    /// Build a synthesis `Problem` from a registry-derived requirement (universal entry).
    pub fn problem_from_requirement(
        &self,
        req: &SynthesisRequirement,
        fn_name: Option<&str>,
    ) -> Result<Problem, BridgeError> {
        let all_examples = synthesis_requirement_to_examples(req)?;
        let name = fn_name.unwrap_or(&req.function_name).to_string();

        // FIX B (single-op fresh holdouts): when the registry op carries enough
        // spec — >=3 DISTINCT example rows — reserve one distinct row as a HELD-OUT
        // generalization probe the solver never sees, mirroring the pipeline path's
        // fresh-holdout differential check. `verify_problem_code_strict` then runs
        // the candidate against this unseen, registry-labelled row, so a program
        // that merely memorised the seed rows fails. (The >=2-example floor in
        // `unsound_confident_solve` already refuses ops too thin to do this; here
        // we keep >=2 seed rows so synthesis itself is not starved.) Below 3
        // distinct rows we keep all as seed (holdouts empty) — examples-only, but
        // the floor has already blocked the dangerous single-row overfit.
        let mut seed: Vec<Example> = Vec::new();
        let mut holdouts: Vec<Example> = Vec::new();
        {
            let mut distinct: Vec<Example> = Vec::new();
            for ex in &all_examples {
                if !distinct.iter().any(|d| d == ex) {
                    distinct.push(ex.clone());
                }
            }
            if distinct.len() >= 3 {
                // Last distinct row → holdout; everything else (incl. duplicates of
                // remaining rows) stays as seed so the solver keeps full signal.
                let reserved = distinct.last().cloned();
                if let Some(reserved) = reserved {
                    for ex in &all_examples {
                        if *ex == reserved && holdouts.is_empty() {
                            holdouts.push(ex.clone());
                        } else {
                            seed.push(ex.clone());
                        }
                    }
                } else {
                    seed = all_examples.clone();
                }
            } else {
                seed = all_examples.clone();
            }
        }
        let examples = seed;
        let signature = infer_signature(&name, &examples);
        let signature = Box::leak(signature.into_boxed_str());
        let category = Box::leak(req.category.clone().into_boxed_str());
        let description = Box::leak(req.description.clone().into_boxed_str());
        Ok(Problem {
            name,
            category,
            description,
            signature,
            examples,
            holdouts,
            reference_code: "",
            synthetic_args: Vec::new(),
            synthetic_values: Vec::new(),
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: Vec::new(),
        })
    }

    /// NL description → verified synthesis via registry requirements (no keyword routing).
    /// NL → verified program. Tries the symbolic comprehension first; if it
    /// produces no successful result AND a local LLM is configured, falls back to
    /// the untrusted LLM lane (translate to a known op / canonical rephrase →
    /// strict-verify). The fallback is INERT without `NSYNTH_LOCAL_LLM_URL` (so
    /// default/CI behavior is unchanged) and recursion-guarded.
    pub fn synthesize_from_description(
        &self,
        description: &str,
        fn_name: Option<&str>,
    ) -> Result<crate::solver::SolveResult, String> {
        let symbolic = self.synthesize_from_description_symbolic(description, fn_name);
        if matches!(&symbolic, Ok(r) if r.success) {
            return symbolic;
        }
        if !IN_LLM_FALLBACK.with(|f| f.get()) {
            if let Some(r) = self.synthesize_via_local_llm(description) {
                return Ok(r);
            }
        }
        symbolic
    }

    /// Symbolic-only NL synthesis (NO LLM fallback). Public so the recall
    /// benchmark can isolate the symbolic baseline from the +LME lane.
    pub fn synthesize_from_description_symbolic(
        &self,
        description: &str,
        fn_name: Option<&str>,
    ) -> Result<crate::solver::SolveResult, String> {
        // COMPOSITIONAL path first: a request that emergently names a two-op
        // array pipeline is comprehensible even though the fail-closed gate would
        // downgrade it (the second op would look "dropped"). Build+verify it here.
        if let Some(outcome) = self.try_compose_pipeline(description) {
            return outcome.map(|o| o.into_solve_result());
        }
        let req = match self.nl_to_requirement(description) {
            Ok(req) => req,
            Err(refusal) => {
                // PHRASE fallback: per-token comprehension refused, but the prose
                // may name a MULTI-WORD op by its own lemma words ("reverse the
                // string" → reverse_string) or a derived phrase surface. Resolve
                // at phrase level and synthesize that REGISTRY op through the
                // trusted example path — strict-verified, fail-closed preserved
                // (a phrase miss falls through to the original refusal).
                if let Some((op, _score)) = self.resolve_phrase_op(description) {
                    if let Some(r) = self.synthesize_op_by_name(&op) {
                        if r.success {
                            return Ok(r);
                        }
                    }
                }
                match refusal {
                    BridgeError::ClarificationNeeded { questions, .. } => {
                        return Err(format_clarification_prompt(&questions));
                    }
                    e => return Err(e.to_string()),
                }
            }
        };
        let problem = self
            .problem_from_requirement(&req, fn_name)
            .map_err(|e| e.to_string())?;
        Ok(solve_verifying_holdouts(&problem))
    }

    /// The synthesizable op names (default_fn_name of every Function entity) —
    /// derived from the LIVE registry, NOT a hand list. Used as the menu the local
    /// LLM translator must pick from.
    pub fn known_op_names(&self) -> Vec<String> {
        let Ok(registry) = self.registry_clone() else {
            return Vec::new();
        };
        let mut ops: Vec<String> = registry
            .get_by_type(&linguigenesis_core::entity::EntityType::Function)
            .into_iter()
            .filter_map(|e| e.get_property("default_fn_name").cloned())
            .collect();
        ops.sort();
        ops.dedup();
        ops
    }

    /// Each known op paired with its registry gloss (first definition). The gloss
    /// menu lets the tiny model match a paraphrase ("accumulate the values") to the
    /// EXACT registered op name (`array_sum`, not the near-miss `sum`) — bare names
    /// alone leave the model guessing what each identifier means.
    pub fn known_op_glosses(&self) -> Vec<(String, String)> {
        let Ok(registry) = self.registry_clone() else {
            return Vec::new();
        };
        let mut ops: Vec<(String, String)> = registry
            .get_by_type(&linguigenesis_core::entity::EntityType::Function)
            .into_iter()
            .filter_map(|e| {
                let name = e.get_property("default_fn_name").cloned()?;
                let gloss = e.definitions.first().cloned().unwrap_or_default();
                Some((name, gloss))
            })
            .collect();
        ops.sort();
        ops.dedup();
        ops
    }

    /// UNTRUSTED local-LLM front door (gated by `NSYNTH_LOCAL_LLM_URL`): translate
    /// arbitrary NL prose → a KNOWN op via a tiny local model, then synthesize that
    /// op through the TRUSTED path (its registry example_cases → solve →
    /// strict-verify). Returns a VERIFIED result only, or `None` when the LLM is
    /// disabled / unsure / picks no known op. The LLM never emits code and never
    /// bypasses verification — it only widens phrasing coverage over the existing
    /// op vocabulary (e.g. "add up all the elements of an array" → array_sum, which
    /// the symbolic comprehension mis-resolves to scalar add).
    pub fn synthesize_via_local_llm(&self, request: &str) -> Option<crate::solver::SolveResult> {
        // Server reachable (or auto-started when NSYNTH_LOCAL_LLM_AUTOSERVE is set);
        // bail fast otherwise so the lane stays inert when no model is available.
        if !crate::local_llm::ensure_server() {
            return None;
        }
        let ops = self.known_op_glosses();
        if ops.is_empty() {
            return None;
        }
        // Mode A — single KNOWN op (safest: only a real op can pass).
        if let Some(op) = crate::local_llm::translate_op(request, &ops) {
            if let Some(r) = self.synthesize_op_by_name(&op) {
                return Some(r);
            }
        }
        // Mode A' — composition breadth: rephrase to canonical NL, then run the
        // EXISTING comprehension (which recognizes filter/map/reduce pipelines) +
        // strict-verify. A bad rephrase fails closed (no verified program). The
        // guard keeps the inner synthesize_from_description on the SYMBOLIC path.
        if let Some(canon) = crate::local_llm::canonical_rephrase(request) {
            IN_LLM_FALLBACK.with(|f| f.set(true));
            let res = self.synthesize_from_description(&canon, None);
            IN_LLM_FALLBACK.with(|f| f.set(false));
            if let Some(r) = res.ok().filter(|r| r.success) {
                return Some(r);
            }
        }
        // Mode B — out-of-vocab (separately gated, RISKIER): LLM-proposed examples.
        self.synthesize_via_llm_examples(request)
    }

    /// Mode B (gated by `NSYNTH_LOCAL_LLM_EXAMPLES`, RISKIER): out-of-vocab —
    /// the LLM proposes I/O examples; synthesize from them with a HELD-OUT
    /// generalization probe (the program must match LLM examples it didn't fit,
    /// catching an inconsistent spec) + strict-verify. UNTRUSTED: a consistently-
    /// wrong LLM spec yields a wrong-but-verified program, so this is a separate
    /// opt-in beyond the basic op/rephrase lane.
    pub fn synthesize_via_llm_examples(&self, request: &str) -> Option<crate::solver::SolveResult> {
        if std::env::var("NSYNTH_LOCAL_LLM_EXAMPLES")
            .ok()
            .filter(|s| !s.is_empty())
            .is_none()
        {
            return None;
        }
        let proposed = crate::local_llm::propose_examples(request)?;
        let json_to_value = |v: &serde_json::Value| -> Option<crate::benchmark::Value> {
            use crate::benchmark::Value;
            if let Some(b) = v.as_bool() {
                return Some(Value::Bool(b));
            }
            if let Some(i) = v.as_i64() {
                return Some(Value::Int(i));
            }
            if let Some(arr) = v.as_array() {
                let ints: Option<Vec<i64>> = arr.iter().map(|x| x.as_i64()).collect();
                return Some(Value::int_array(&ints?));
            }
            v.as_str().map(|s| Value::Str(s.to_string()))
        };
        let mut exs: Vec<crate::benchmark::Example> = Vec::new();
        for p in &proposed {
            let inputs: Option<Vec<_>> = p.inputs.iter().map(&json_to_value).collect();
            let (Some(inputs), Some(output)) = (inputs, json_to_value(&p.output)) else {
                continue;
            };
            exs.push(crate::benchmark::Example { inputs, expected: output });
        }
        // Reject an inconsistent spec (same input -> two outputs) and dedup exact
        // repeats, so the held-out probe below uses genuinely DISTINCT inputs (a
        // repeated example would make the holdout vacuous).
        let exs = crate::benchmark::dedup_consistent_examples(&exs)?;
        if exs.len() < 4 {
            return None;
        }
        // HELD-OUT guard: reserve the last 2 examples as a generalization probe the
        // solver never fits. NOTE: `solve_problem` solves a `synthesis_view()` that
        // CLEARS `problem.holdouts`, so the probe does NOT bite inside the solver —
        // we MUST re-verify the solved code against the holdouts HERE (post-solve),
        // or an overfit-to-seed program that contradicts the rest of the LLM's spec
        // would pass. `code_reproduces_examples` is that real re-check.
        let split = exs.len().saturating_sub(2).max(2);
        let (seed, holdouts) = exs.split_at(split);
        let name = "f".to_string();
        let signature: &'static str = Box::leak(infer_signature(&name, seed).into_boxed_str());
        let problem = crate::benchmark::Problem {
            name,
            category: "local-llm-examples",
            description: "llm-proposed examples",
            signature,
            examples: seed.to_vec(),
            holdouts: holdouts.to_vec(),
            ..Default::default()
        };
        let res = crate::solver::solve_problem(&problem);
        // Re-verify the synthesized code against the FULL spec (seed + holdouts) at
        // the interpreter level: holdouts are the generalization probe solve_problem
        // strips, and re-checking the seed too catches a search/codegen mismatch
        // (the search claims a seed fit but the emitted code executes differently).
        if !res.success || !crate::runtime::code_reproduces_examples(&res.code, &exs) {
            return None;
        }
        Some(res)
    }

    /// Mode C (project decomposition, gated by `NSYNTH_LOCAL_LLM_PROJECT`): the
    /// untrusted LLM breaks an open-ended request into named sub-functions; each is
    /// synthesized through the NORMAL door (`synthesize_from_description`, which
    /// strict-verifies and itself auto-falls-back to the LLM lane). Returns
    /// `(verified, failed)`: the verified `(name, result)` components and the names
    /// the engine could NOT verify.
    ///
    /// TRUST: the LLM only proposes the decomposition; every returned component is
    /// strict-verified. What is NOT verified is the WHOLE-ARTIFACT behavior — there
    /// is no example oracle for "does the assembled program do what was asked". So
    /// this delivers *verified parts of a plausible plan*, not a verified program.
    pub fn synthesize_project_via_llm(
        &self,
        request: &str,
    ) -> Option<(Vec<(String, crate::solver::SolveResult)>, Vec<String>)> {
        if std::env::var("NSYNTH_LOCAL_LLM_PROJECT")
            .ok()
            .filter(|s| !s.is_empty())
            .is_none()
        {
            return None;
        }
        if !crate::local_llm::ensure_server() {
            return None;
        }
        let components = crate::local_llm::propose_decomposition(request)?;
        let mut verified = Vec::new();
        let mut failed = Vec::new();
        let mut used: std::collections::HashSet<String> = std::collections::HashSet::new();
        for comp in &components {
            // Collision-free fn name (mirrors synthesize_project's naming).
            let base = if comp.name.is_empty() { "f".to_string() } else { comp.name.clone() };
            let mut name = base.clone();
            let mut n = 2;
            while used.contains(&name) {
                name = format!("{base}{n}");
                n += 1;
            }
            used.insert(name.clone());
            match self.synthesize_from_description(&comp.description, Some(&name)) {
                Ok(r) if r.success => verified.push((name, r)),
                _ => failed.push(format!("{name}: {}", comp.description)),
            }
        }
        Some((verified, failed))
    }

    /// Mode D (verify-and-repair, gated by `NSYNTH_LOCAL_LLM_REPAIR`): the LLM writes
    /// a WHOLE Mog program for a task no known op / example-search can produce; the
    /// engine RUNS it against every example and, on failure, feeds the concrete
    /// failure back for a fix — iterating up to `NSYNTH_LOCAL_LLM_REPAIR_TRIES`
    /// (default 4). Accepts ONLY a program that reproduces EVERY example, so a wrong
    /// program never passes (the verification guarantee a raw model lacks). This is
    /// the lever that scales to arbitrary algorithms; `request` is the task
    /// description, and the examples are appended to the prompt automatically.
    pub fn synthesize_via_repair_loop(
        request: &str,
        examples: &[crate::benchmark::Example],
    ) -> Option<crate::solver::SolveResult> {
        if std::env::var("NSYNTH_LOCAL_LLM_REPAIR")
            .ok()
            .filter(|s| !s.is_empty())
            .is_none()
        {
            return None;
        }
        if examples.is_empty() || !crate::local_llm::ensure_server() {
            return None;
        }
        let tries: usize = std::env::var("NSYNTH_LOCAL_LLM_REPAIR_TRIES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);
        // Show the model the concrete examples (a bounded sample) so it targets the
        // exact contract, not just the prose.
        let ex_str = examples
            .iter()
            .take(6)
            .map(|e| format!("  input {:?} -> output {:?}", e.inputs, e.expected))
            .collect::<Vec<_>>()
            .join("\n");
        // Pin the EXACT signature (inferred from the examples) so the model fills only
        // the body — it was otherwise inventing extra params from the grammar example.
        let signature = infer_signature("solve", examples);
        let full_request = format!(
            "{request}\n\nUse EXACTLY this signature (fill the body):\n{signature}\n\n\
             The function must satisfy these examples:\n{ex_str}"
        );

        // BEST-OF-N + repair: each round draws N candidates (a greedy anchor at
        // temp 0 + N-1 diverse samples at temp 0.8) and the VERIFIER keeps the first
        // that reproduces every example — pass@N >> pass@1 for a small model, and
        // verification makes the extra samples free of risk. If none pass, the greedy
        // anchor's failure drives the next round's repair. N = NSYNTH_LOCAL_LLM_SAMPLES.
        let samples: usize = std::env::var("NSYNTH_LOCAL_LLM_SAMPLES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4)
            .max(1);
        let verified = |code: String| {
            Some(crate::solver::SolveResult {
                success: true,
                code,
                method: "llm-repair".to_string(),
                error: None,
                metadata: Default::default(),
            })
        };
        let mut prior: Option<(String, String)> = None;
        for _ in 0..tries {
            let prior_ref = prior.as_ref().map(|(c, e)| (c.as_str(), e.as_str()));
            // Greedy anchor (temp 0): the canonical attempt + the repair seed.
            let Some(anchor) = crate::local_llm::propose_program(&full_request, prior_ref, 0.0)
            else {
                break;
            };
            let anchor_failure = match crate::runtime::describe_first_failure(&anchor, examples) {
                None => return verified(anchor),
                Some(f) => f,
            };
            // N-1 diverse samples (temp 0.8) — breadth; the verifier filters.
            for _ in 1..samples {
                if let Some(cand) =
                    crate::local_llm::propose_program(&full_request, prior_ref, 0.8)
                {
                    if crate::runtime::describe_first_failure(&cand, examples).is_none() {
                        return verified(cand);
                    }
                }
            }
            {
                let failure = anchor_failure;
                let code = anchor;
                {
                    eprintln!("[repair] retry: {}", &failure[..failure.len().min(80)]);
                    prior = Some((code, failure));
                }
            }
        }
        None
    }

    /// Mode C+ (contract-driven project synthesis, gated by `NSYNTH_LOCAL_LLM_PROJECT`):
    /// the untrusted LLM proposes a decomposition WHERE EACH COMPONENT CARRIES ITS OWN
    /// I/O EXAMPLES. Each component's examples become a real `Problem` (seed +
    /// held-out probe) that `solve_problem` strict-verifies; only verified components
    /// are returned (with the FULL example set, so a downstream writer can emit
    /// reproduction tests). Returns `(verified, failed)`.
    ///
    /// TRUST: the LLM only proposes the decomposition AND the examples; every
    /// returned component is strict-verified against (a held-out split of) those
    /// examples, so a component that merely memorises a wrong seed fails closed. What
    /// remains UNVERIFIED is the WHOLE-ARTIFACT behavior — there is no oracle for
    /// "does the assembled program do what was asked", and the examples themselves
    /// are the LLM's unverified claim. So this delivers *verified parts of a
    /// plausible plan*, not a verified program.
    pub fn synthesize_project_with_contracts(
        &self,
        request: &str,
    ) -> Option<(Vec<VerifiedComponent>, Vec<String>)> {
        if std::env::var("NSYNTH_LOCAL_LLM_PROJECT")
            .ok()
            .filter(|s| !s.is_empty())
            .is_none()
        {
            return None;
        }
        if !crate::local_llm::ensure_server() {
            return None;
        }
        let specs = crate::local_llm::propose_decomposition_with_contracts(request)?;
        let mut verified = Vec::new();
        let mut failed = Vec::new();
        let mut used: std::collections::HashSet<String> = std::collections::HashSet::new();
        for spec in &specs {
            // Collision-free fn name (mirrors synthesize_project_via_llm's naming).
            let base = if spec.name.is_empty() { "f".to_string() } else { spec.name.clone() };
            let mut name = base.clone();
            let mut n = 2;
            while used.contains(&name) {
                name = format!("{base}{n}");
                n += 1;
            }
            used.insert(name.clone());

            // Map each proposed example to a runtime Example; skip any that don't map.
            let mut exs: Vec<crate::benchmark::Example> = Vec::new();
            for p in &spec.examples {
                let inputs: Option<Vec<_>> = p.inputs.iter().map(json_to_bench_value).collect();
                let (Some(inputs), Some(output)) = (inputs, json_to_bench_value(&p.output)) else {
                    continue;
                };
                exs.push(crate::benchmark::Example { inputs, expected: output });
            }
            // Reject an inconsistent spec (same input -> two outputs) and dedup exact
            // repeats so the held-out probe uses distinct inputs.
            let Some(exs) = crate::benchmark::dedup_consistent_examples(&exs) else {
                failed.push(format!("{name}: inconsistent examples (same input, different output)"));
                continue;
            };
            // Need >=3 DISTINCT mappable examples so the proven Mode-B split keeps a
            // non-empty seed AND a held-out probe. No description fallback: a contract
            // lane without real examples cannot be strict-verified or reproduced.
            if exs.len() < 3 {
                failed.push(format!("{name}: {}", spec.description));
                continue;
            }
            // HELD-OUT guard (same formula as synthesize_via_llm_examples): reserve
            // the last <=2 examples as a generalization probe. solve_problem strips
            // problem.holdouts via synthesis_view, so the probe is re-checked
            // post-solve with code_reproduces_examples (else it would be cosmetic).
            let split = exs.len().saturating_sub(2).max(2);
            let (seed, holdouts) = exs.split_at(split);
            let signature: &'static str = Box::leak(infer_signature(&name, seed).into_boxed_str());
            let problem = crate::benchmark::Problem {
                name: name.clone(),
                category: "local-llm-contracts",
                description: "llm-proposed contract",
                signature,
                examples: seed.to_vec(),
                holdouts: holdouts.to_vec(),
                ..Default::default()
            };
            let res = crate::solver::solve_problem(&problem);
            // Re-verify against the FULL spec (seed + holdouts) at the interpreter
            // level (holdouts = the generalization probe solve_problem strips; the
            // seed re-check catches a search/codegen mismatch).
            if res.success && crate::runtime::code_reproduces_examples(&res.code, &exs) {
                // Store the FULL example set (seed + holdouts) for reproduction tests.
                verified.push(VerifiedComponent { name, result: res, examples: exs });
            } else {
                failed.push(format!("{name}: {}", spec.description));
            }
        }
        Some((verified, failed))
    }

    /// Synthesize a KNOWN op directly from its registry `example_cases` (its
    /// TRUSTED spec) — bypassing NL comprehension, since the input is the op's
    /// own name. solve_problem strict-verifies; returns a verified result or None.
    pub fn synthesize_op_by_name(&self, op_name: &str) -> Option<crate::solver::SolveResult> {
        use linguigenesis_core::coding_requirements::{parse_example_cases, LiteralValue};
        use linguigenesis_core::entity::EntityType;
        let registry = self.registry_clone().ok()?;
        let entity = registry
            .get_by_type(&EntityType::Function)
            .into_iter()
            .find(|e| e.get_property("default_fn_name").map(|n| n == op_name).unwrap_or(false))?;
        let specs = parse_example_cases(&entity);
        if specs.len() < 2 {
            return None;
        }
        // Struct fields convert recursively through the shared converter.
        // Struct fields convert recursively through the shared converter.
        let lit = |l: &LiteralValue| -> crate::benchmark::Value {
            literal_to_value(l).unwrap_or(crate::benchmark::Value::Int(0))
        };
        let examples: Vec<crate::benchmark::Example> = specs
            .iter()
            .map(|s| crate::benchmark::Example {
                inputs: s.inputs.iter().map(&lit).collect(),
                expected: lit(&s.expected),
            })
            .collect();
        let signature: &'static str =
            Box::leak(infer_signature(op_name, &examples).into_boxed_str());
        let problem = crate::benchmark::Problem {
            name: op_name.to_string(),
            // "registry-op" routes AROUND the library-alias tier: trusted leaves
            // synthesize from first principles (see op_library::try_library).
            category: "registry-op",
            description: "registry op",
            signature,
            examples,
            ..Default::default()
        };
        let res = crate::solver::solve_problem(&problem);
        // The component writer imports `crate::<op>::<op>`, so the winning code
        // must genuinely DEFINE `fn <op_name>` — a tier that emits differently-
        // named fns (e.g. a library pipeline of other ops) would compile-break
        // every downstream consumer. Fail closed instead.
        if res.success && !res.code.contains(&format!("fn {op_name}(")) {
            return None;
        }
        res.success.then_some(res)
    }

    /// COMPOSITIONAL front door: if the request emergently names a two-op
    /// array pipeline `reduce(map(arr))` (or its reduce-only/map-anchored
    /// degenerate), build that pipeline from the resolved primitives, synthesize
    /// it through the EXISTING solver, and strict-verify it on FRESH holdouts
    /// labelled by an independent reference composition. Returns:
    ///   * `Some(Ok(outcome))` — a pipeline was recognised AND accepted,
    ///   * `Some(Err(reason))` — a pipeline was recognised but could not be built
    ///     or did not strict-verify (fail honestly, do NOT silently fall back),
    ///   * `None` — the request is not a two-op pipeline (caller uses single-op).
    ///
    /// No phrase→plan table: linguigenesis-core's `comprehend` EMITS the plan in
    /// `req.pipeline` (see `coding_requirements::classify_pipeline`). The bridge no
    /// longer derives it — it only EXECUTES the comprehended plan.
    pub fn try_compose_pipeline(&self, description: &str) -> Option<Result<PipelineOutcome, String>> {
        let registry = match self.registry_clone() {
            Ok(r) => r,
            Err(e) => return Some(Err(e.to_string())),
        };
        // Comprehend the request; linguigenesis-core decides — purely from registry
        // signatures — whether it is a composable pipeline and, if so, populates
        // `req.pipeline`. Single-op (and inline-example) requests come back with
        // `pipeline = None`, so the caller falls through to the single-op door.
        let mut coding = CodingComprehension::new(registry);
        let req = coding.comprehend(description);
        let plan = req.pipeline.clone()?;
        Some(self.build_and_verify_pipeline(description, &plan))
    }

    fn build_and_verify_pipeline(
        &self,
        description: &str,
        plan: &CompositionPlan,
    ) -> Result<PipelineOutcome, String> {
        // 1. Synthesize each primitive through the EXISTING solver from its
        //    registry example_cases. Each map op in the CHAIN defines one element
        //    transform; the optional reduce primitive defines the fold.
        //    `map_chain` is in REQUEST ORDER (outer→inner).
        let mut map_chain: Vec<(String, String)> = Vec::with_capacity(plan.maps.len());
        for m in &plan.maps {
            map_chain.push((m.fn_name.clone(), self.synthesize_primitive(m)?));
        }

        // 2. Classify the reduce fold (if any) by EXECUTING the synthesized reduce
        //    code on probe inputs (behaviour-driven; never keyed on the op's name).
        //    An ArrayReduce is probed with arrays; a BinaryFoldSeed (e.g. add) with
        //    scalar pairs — the shape comes from its emergent `op_role`.
        let fold = match &plan.reduce {
            Some(r) => {
                let reduce_code = self.synthesize_primitive(r)?;
                let fk = match r.role {
                    OpRole::ArrayReduce => classify_array_fold(&reduce_code, &r.fn_name),
                    OpRole::BinaryFoldSeed => classify_binary_fold(&reduce_code, &r.fn_name),
                    _ => None,
                }
                .ok_or_else(|| {
                    format!("could not classify fold for reduce op '{}'", r.fn_name)
                })?;
                Some(fk)
            }
            None => None,
        };

        // 2a-FILTER. A predicate filter ("the positive values", "sum of the even
        //     values") is its own pipeline shape: map-chain → filter → optional
        //     Sum/Product reduce. Mutually exclusive with an array transform
        //     (enforced in classify_pipeline). Max/Min over a filtered (possibly
        //     empty) array is edge-casey → deferred (falls back to non-filter).
        if let Some(fp) = &plan.filter {
            if matches!(fold, Some(FoldKind::Max) | Some(FoldKind::Min)) {
                return Err("filter + max/min reduce not yet supported".to_string());
            }
            let composed_name = pipeline_fn_name(plan);
            let reference = emit_filter_pipeline_reference(&composed_name, fold, fp, &map_chain);
            let reference: &'static str = Box::leak(reference.into_boxed_str());
            let ret = if fold.is_some() { "i64" } else { "[i64]" };
            let signature: &'static str = Box::leak(
                format!("fn {}(a: [i64]) -> {}", composed_name, ret).into_boxed_str(),
            );
            let mut problem = crate::benchmark::problem_from_reference(
                &composed_name,
                signature,
                reference,
            )
            .map_err(|e| format!("filter pipeline reference unrunnable: {e}"))?;
            problem.category = Box::leak("nl-compose".to_string().into_boxed_str());
            problem.description = Box::leak(
                format!("filter pipeline for: {description}").into_boxed_str(),
            );
            let solved = crate::solver::solve_problem(&problem);
            if !solved.success {
                return Err(format!(
                    "filter pipeline ({}) recognised but solver could not synthesize it (method={}, err={:?})",
                    describe_plan(plan),
                    solved.method,
                    solved.error
                ));
            }
            crate::runtime::verify_problem_code_strict(&problem, &solved.code).map_err(|e| {
                format!(
                    "filter pipeline OVERFIT — strict holdout verification failed: {e}\nCODE:\n{}",
                    solved.code
                )
            })?;
            return Ok(PipelineOutcome {
                description: description.to_string(),
                fn_name: composed_name.clone(),
                map_fns: plan.maps.iter().map(|m| m.fn_name.clone()).collect(),
                array_xfm_fns: Vec::new(),
                array_xfms: Vec::new(),
                reduce_fn: plan.reduce.as_ref().map(|r| r.fn_name.clone()),
                fold,
                code: solved.code,
                method: format!("nl-compose-filter:{}", solved.method),
            });
        }

        // 2b. Classify the array transform (if any) behaviourally — never by name.
        //     First try EXECUTING the synthesized transform code on probe arrays;
        //     if that is inconclusive (a single-example registry op can be
        //     affine-overfit by the solver — e.g. reverse's lone `[1,2,3]->[3,2,1]`
        //     fits `y=-x+4`), fall back to matching the op's REGISTRY example_cases
        //     (its verified output spec) against the candidate transforms. Both
        //     paths are output-grounded, not name-keyed.
        let mut array_xfms: Vec<ArrayTransformKind> =
            Vec::with_capacity(plan.array_transforms.len());
        for t in &plan.array_transforms {
            let by_exec = self
                .synthesize_primitive(t)
                .ok()
                .and_then(|code| classify_array_transform_by_exec(&code, &t.fn_name));
            let kind = match by_exec {
                Some(k) => k,
                None => self.classify_array_transform_by_spec(t).ok_or_else(|| {
                    format!(
                        "could not classify array transform '{}' as sort/reverse \
                         (neither execution nor registry example_cases matched)",
                        t.fn_name
                    )
                })?,
            };
            array_xfms.push(kind);
        }

        // 3. Emit the composed REFERENCE: the map fn bodies (chained), then the
        //    optional array transform on the built array, then either a fused fold
        //    driver (shape a, scalar out) or the array itself (shape b, array out).
        //    This is an INDEPENDENT implementation of the pipeline, used only to
        //    LABEL fresh holdouts.
        let composed_name = pipeline_fn_name(plan);
        let reference = emit_pipeline_reference(&composed_name, fold, &array_xfms, &map_chain);
        let reference: &'static str = Box::leak(reference.into_boxed_str());
        // Scalar output when a reduce is present; array output for a pure map chain.
        let ret = if fold.is_some() { "i64" } else { "[i64]" };
        let signature: &'static str =
            Box::leak(format!("fn {}(a: [i64]) -> {}", composed_name, ret).into_boxed_str());

        // 4. Build a Problem whose reference IS the composition, so the existing
        //    strict verifier samples FRESH arrays and labels them by RUNNING the
        //    reference. Seed examples are likewise produced by the reference.
        let mut problem = crate::benchmark::problem_from_reference(
            &composed_name,
            signature,
            reference,
        )
        .map_err(|e| format!("pipeline reference unrunnable: {e}"))?;
        let category: &'static str = Box::leak("nl-compose".to_string().into_boxed_str());
        let descr: &'static str =
            Box::leak(format!("transform-chain pipeline for: {description}").into_boxed_str());
        problem.category = category;
        problem.description = descr;

        // 5. Synthesize the WHOLE pipeline through the existing solver from the
        //    seed examples (the array engine handles map/reduce per U5a/U5b and
        //    array→array transforms per array_transform).
        let solved = crate::solver::solve_problem(&problem);
        if !solved.success {
            return Err(format!(
                "transform-chain pipeline recognised ({}) but solver could not synthesize it (method={}, err={:?})",
                describe_plan(plan),
                solved.method,
                solved.error
            ));
        }

        // 6. STRICT verification on FRESH holdouts (reference-labelled): the solved
        //    program must match the independent composition on unseen arrays.
        crate::runtime::verify_problem_code_strict(&problem, &solved.code).map_err(|e| {
            format!(
                "transform-chain pipeline OVERFIT — strict holdout verification failed: {e}\nCODE:\n{}",
                solved.code
            )
        })?;

        Ok(PipelineOutcome {
            description: description.to_string(),
            fn_name: composed_name.clone(),
            map_fns: plan.maps.iter().map(|m| m.fn_name.clone()).collect(),
            array_xfm_fns: plan.array_transforms.iter().map(|t| t.fn_name.clone()).collect(),
            array_xfms,
            reduce_fn: plan.reduce.as_ref().map(|r| r.fn_name.clone()),
            fold,
            code: solved.code,
            method: format!("nl-compose-chain:{}", solved.method),
        })
    }

    /// Synthesize a single registry primitive (map or reduce op) through the
    /// existing solver, returning its verified code. The primitive is described
    /// by its registry entity's `example_cases`, so this is the same proven path
    /// `every_registry_operation_is_synthesizable` exercises.
    fn synthesize_primitive(&self, op: &OpRef) -> Result<String, String> {
        use linguigenesis_core::entity::EntityType;
        let registry = self.registry_clone().map_err(|e| e.to_string())?;
        let entity = registry
            .get_by_type(&EntityType::Function)
            .into_iter()
            .find(|e| {
                e.get_property("default_fn_name")
                    .map(|f| f == &op.fn_name)
                    .unwrap_or(false)
                    || e.lemma == op.fn_name
            })
            .ok_or_else(|| format!("primitive '{}' not found in registry", op.fn_name))?;
        let req = SynthesisRequirement::from_operation_entity(&entity)
            .ok_or_else(|| format!("primitive '{}' is not synthesizable", op.fn_name))?;
        let result = self
            .synthesize_from_requirement(&req, Some(&req.function_name))
            .map_err(|e| format!("primitive '{}' synthesis: {e}", op.fn_name))?;
        if !result.success {
            return Err(format!(
                "primitive '{}' did not synthesize (method={}, err={:?})",
                op.fn_name, result.method, result.error
            ));
        }
        Ok(result.code)
    }

    /// Classify an ArrayTransform op by matching its REGISTRY `example_cases` (the
    /// op's verified output spec) against each candidate transform. This is the
    /// output-grounded fallback for when the synthesized primitive is an affine
    /// overfit of a single example (so executing it is inconclusive). It reads the
    /// op's labelled (input-array → output-array) pairs from the registry and asks
    /// which of {sort, reverse} reproduces EVERY pair — never the op's name.
    fn classify_array_transform_by_spec(&self, op: &OpRef) -> Option<ArrayTransformKind> {
        use linguigenesis_core::coding_requirements::LiteralValue;
        use linguigenesis_core::entity::EntityType;
        let registry = self.registry_clone().ok()?;
        let entity = registry.get_by_type(&EntityType::Function).into_iter().find(|e| {
            e.get_property("default_fn_name")
                .map(|f| f == &op.fn_name)
                .unwrap_or(false)
                || e.lemma == op.fn_name
        })?;
        let req = SynthesisRequirement::from_operation_entity(&entity)?;
        // Collect (input array, output array) pairs from the op's example_cases.
        let mut pairs: Vec<(Vec<i64>, Vec<i64>)> = Vec::new();
        for spec in &req.examples {
            let (Some(LiteralValue::Array(inp)), LiteralValue::Array(out)) =
                (spec.inputs.first(), &spec.expected)
            else {
                return None; // not an array→array op spec
            };
            pairs.push((inp.clone(), out.clone()));
        }
        if pairs.is_empty() {
            return None;
        }
        for cand in [ArrayTransformKind::Sort, ArrayTransformKind::Reverse] {
            let all = pairs.iter().all(|(inp, out)| {
                let mut expected = inp.clone();
                match cand {
                    ArrayTransformKind::Sort => expected.sort(),
                    ArrayTransformKind::Reverse => expected.reverse(),
                }
                &expected == out
            });
            if all {
                return Some(cand);
            }
        }
        None
    }

    /// Synthesize from an already-derived requirement (e.g. after clarification).
    pub fn synthesize_from_requirement(
        &self,
        req: &SynthesisRequirement,
        fn_name: Option<&str>,
    ) -> Result<crate::solver::SolveResult, String> {
        let problem = self
            .problem_from_requirement(req, fn_name)
            .map_err(|e| e.to_string())?;
        Ok(solve_verifying_holdouts(&problem))
    }

    /// Structural multi-component signal: does this request DESCRIBE >=2 component
    /// functions (i.e. a PROJECT) rather than a single (possibly compositional)
    /// function? Reuses the EXACT same `comprehend_project` decomposition
    /// (`split_component_clauses`, function-head count) that `synthesize_project`
    /// uses to split — there is NO separate heuristic and NO phrase table. A bare
    /// composition ("the larger of two numbers then triples it", 0 heads) and a
    /// single described function ("a function that ... then ...", 1 head) both
    /// yield a 1-component plan → `false`; only a request with >=2 function heads
    /// ("a module with a function that ..., and a function that ...") → `true`.
    ///
    /// The CLI front door (`session::handle_query`) consults this to keep a
    /// multi-component request OUT of the single-function compositional intercept
    /// so it reaches the `synthesize_project` multi-file door instead.
    pub fn is_multi_component(&self, text: &str) -> bool {
        let Ok(registry) = self.registry_clone() else {
            return false;
        };
        let mut coding = CodingComprehension::new(registry);
        coding.comprehend_project(text).components.len() >= 2
    }

    /// Comprehend a (possibly multi-component) request into a `ProjectPlan` and
    /// synthesize each component INDEPENDENTLY through the existing single-op
    /// door (`problem_from_requirement` + `solve_problem`, via
    /// `synthesize_from_requirement`). A single-function request yields a
    /// 1-element vector — identical to today's single-file path. Components
    /// whose requirement carries no examples (did not comprehend) are SKIPPED
    /// and reported in the returned skip list rather than fabricated.
    ///
    /// Returns `(Vec<(fn_name, SolveResult)>, Vec<skipped_reason>)`. The split
    /// itself lives in linguigenesis-core (`comprehend_project`); the bridge only
    /// loops + solves. No new synthesis path.
    pub fn synthesize_project(
        &self,
        text: &str,
    ) -> Result<(Vec<(String, crate::solver::SolveResult)>, Vec<String>), String> {
        let registry = self.registry_clone().map_err(|e| e.to_string())?;
        let mut coding = CodingComprehension::new(registry);
        let plan: ProjectPlan = coding.comprehend_project(text);

        // Assign each component a stable, collision-free fn/module name up-front,
        // so a CONSUMER's composed-example derivation and the writer's
        // use-injection agree on the producer's name. (The single-function and
        // independent-sibling cases keep the exact names they had before.)
        let mut used_names: std::collections::HashSet<String> = std::collections::HashSet::new();
        let names: Vec<String> = plan
            .components
            .iter()
            .map(|req| {
                let base = if req.function_name.is_empty() {
                    "f".to_string()
                } else {
                    req.function_name.clone()
                };
                let mut name = base.clone();
                let mut n = 2;
                while used_names.contains(&name) {
                    name = format!("{base}{n}");
                    n += 1;
                }
                used_names.insert(name.clone());
                name
            })
            .collect();

        // EMERGENT inter-component edges: `deps[i] = (consumer, producer)`. For
        // each consumer that depends on exactly one producer, solve it through the
        // COMPOSED-example + Call-node door (see `synthesize_consumer_with_call`).
        // A component that is the CONSUMER of such an edge is skipped in the plain
        // loop below (it is solved by the dep path). Independent / single-function
        // requests have `deps == []`, so the plain loop runs unchanged.
        let mut producer_of: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        for &(c, p) in &plan.deps {
            // MVP: a single producer per consumer (single producer→consumer DAG
            // edge). A second producer for the same consumer is ignored here and
            // reported rather than silently mis-composed.
            producer_of.entry(c).or_insert(p);
        }

        let mut solved: Vec<(String, crate::solver::SolveResult)> =
            Vec::with_capacity(plan.components.len());
        let mut skipped = Vec::new();
        // Cache solved producer code by index so a consumer can compose against it.
        let mut solved_code: std::collections::HashMap<usize, String> =
            std::collections::HashMap::new();

        // PASS 1: solve every NON-consumer (producer + independent) component on
        // the existing single-op door. This topo-orders producers before the
        // consumers that call them (every dep edge points consumer→producer, and
        // producers are never themselves consumers in this single-edge MVP).
        // A registry handle for the per-component P2C compositional classifier
        // (the project-comprehension registry was moved into `coding` above).
        let class_registry = self.registry_clone().map_err(|e| e.to_string())?;
        for (idx, req) in plan.components.iter().enumerate() {
            if producer_of.contains_key(&idx) {
                continue; // a consumer — solved in pass 2
            }
            // P2C COMPOSITIONAL ROUTING (BUILD-B-MULTICOMPONENT-DECOMP): a component
            // whose DESCRIBED clause is a scalar `"X then Y"` composition is
            // auto-contracted through the SAME P2C path a single described function
            // uses (`classify_compositional` -> `emit_scalar_reference` ->
            // `problem_from_reference` -> solve + strict-verify) instead of being
            // mis-solved as its HEAD op alone. A single-op clause comes back
            // `NotCompositional` and falls through to the UNCHANGED single-op door
            // below (registry-op multi-file paths stay byte-identical). A confirmed
            // composition whose later atom is unresolvable refuses HONESTLY here —
            // it is reported in `skipped`, never fabricated.
            match crate::reference_nl::classify_compositional(&req.description, &class_registry) {
                crate::reference_nl::CompositionalIntake::Compositional {
                    name: comp_name,
                    signature,
                    chain,
                } => {
                    match self.solve_compositional_component(&comp_name, &signature, &chain) {
                        Ok(result) => {
                            solved_code.insert(idx, result.code.clone());
                            solved.push((comp_name, result));
                        }
                        Err(e) => skipped.push(format!(
                            "component '{}' (compositional) failed: {e}",
                            req.description
                        )),
                    }
                    continue;
                }
                crate::reference_nl::CompositionalIntake::Unresolvable(reason) => {
                    skipped.push(format!(
                        "component '{}' has an unresolvable atom: {reason}",
                        req.description
                    ));
                    continue;
                }
                crate::reference_nl::CompositionalIntake::NotCompositional => {}
            }
            if req.examples.is_empty() {
                skipped.push(format!(
                    "component '{}' did not comprehend (no examples derived)",
                    req.description
                ));
                continue;
            }
            let name = &names[idx];
            match self.synthesize_from_requirement(req, Some(name)) {
                Ok(result) if result.success => {
                    solved_code.insert(idx, result.code.clone());
                    solved.push((name.clone(), result));
                }
                Ok(result) => skipped.push(format!(
                    "component '{}' failed to synthesize: {}",
                    name,
                    result.error.unwrap_or_else(|| "no solution".to_string())
                )),
                Err(e) => skipped.push(format!("component '{name}' error: {e}")),
            }
        }

        // PASS 2: solve each CONSUMER against its producer via the Call-node door.
        for (idx, req) in plan.components.iter().enumerate() {
            let Some(&pidx) = producer_of.get(&idx) else {
                continue;
            };
            let consumer_name = &names[idx];
            let producer_name = &names[pidx];
            let Some(producer_code) = solved_code.get(&pidx).cloned() else {
                skipped.push(format!(
                    "consumer '{consumer_name}' skipped: producer '{producer_name}' did not solve"
                ));
                continue;
            };
            match self.synthesize_consumer_with_call(
                req,
                consumer_name,
                producer_name,
                &plan.components[pidx],
                &producer_code,
            ) {
                Ok(result) => solved.push((consumer_name.clone(), result)),
                Err(e) => skipped.push(format!(
                    "consumer '{consumer_name}' (calls '{producer_name}') skipped: {e}"
                )),
            }
        }
        Ok((solved, skipped))
    }

    /// COMPOSED-EXAMPLE + CALL-NODE door for a consumer B that depends on a sibling
    /// producer A (edge detected emergently in `comprehend_project`). B's request
    /// comprehends to its OWN op's RAW examples (the comprehension confusion the
    /// HARD-MECHANISM note records), so we cannot search B directly. Instead:
    ///
    ///   1. RESIDUAL h: B's clause minus the producer reference. We resolve it as
    ///      B's own resolved op (`req.function_name`) and synthesize it from its
    ///      registry entity through the SAME single-op door producers use. If
    ///      B's op IS the producer (pure alias, `square(x) = square(x)`), h is the
    ///      identity and no residual synthesis is needed.
    ///   2. COMPOSED EXAMPLES: run the SOLVED producer A on A's example inputs and
    ///      apply the SOLVED residual h: `B(x) = h(A(x))`. These are DERIVED by
    ///      execution (never fabricated): the producer + residual are real solved
    ///      programs the runtime evaluates.
    ///   3. CALL SEARCH: register A as a `NamedCallable` (its arity + Mog source +
    ///      `eval`) and solve B's COMPOSED problem via
    ///      `enumerative::synthesize_scalar_with_callees`. The Call-node search
    ///      discovers `A(x)`-bearing expressions (e.g. `negate(square(a))` /
    ///      `square(a) + 1`), and the strict-verify gate prepends A's source so the
    ///      call resolves. The returned code names A — the writer injects the
    ///      `use crate::<A_module>::<A_fn>;` import.
    ///
    /// Only single-arg producers + single-arg residuals (the arities the call wiring
    /// registers) are handled; anything else is reported honestly (no fabrication).
    fn synthesize_consumer_with_call(
        &self,
        consumer_req: &SynthesisRequirement,
        consumer_name: &str,
        producer_name: &str,
        producer_req: &SynthesisRequirement,
        producer_code: &str,
    ) -> Result<crate::solver::SolveResult, String> {
        use crate::enumerative::NamedCallable;

        // The producer must be a single-arg scalar op (the only call arity the
        // composed-example derivation + the search wiring support here).
        let producer_inputs = producer_req
            .examples
            .first()
            .map(|e| e.inputs.len())
            .unwrap_or(0);
        if producer_inputs != 1 {
            return Err(format!(
                "producer '{producer_name}' is not single-arg (arity {producer_inputs}); \
                 single-arg producer→consumer is the MVP"
            ));
        }

        // The producer's example inputs are the x's we compose over. Reuse the
        // producer's OWN comprehended example inputs (real, registry-derived).
        let xs: Vec<i64> = producer_req
            .examples
            .iter()
            .filter_map(|spec| match spec.inputs.first() {
                Some(LiteralValue::Int(v)) => Some(*v),
                _ => None,
            })
            .collect();
        if xs.len() < 2 {
            return Err("producer carries fewer than 2 integer example inputs to compose over".into());
        }

        // RESIDUAL h: identity when B's op IS the producer (pure alias); else the
        // single-arg op B itself resolves to (`consumer_req.function_name`),
        // synthesized from its registry entity through the same single-op door.
        let residual_code: Option<String> = if consumer_req.function_name == producer_name {
            None // pure alias B(x) = A(x)
        } else {
            let code = self.synthesize_named_unary_op(&consumer_req.function_name).map_err(|e| {
                format!(
                    "residual op '{}' could not be derived as a single-arg function: {e}",
                    consumer_req.function_name
                )
            })?;
            Some(code)
        };

        // COMPOSED EXAMPLES: B(x) = h(A(x)), derived by RUNNING the solved
        // producer then the solved residual. No fabrication: every expected value
        // is produced by executing real synthesized code.
        let mut composed: Vec<Example> = Vec::new();
        for &x in &xs {
            let a_out = crate::runtime::execute_function(
                producer_code,
                producer_name,
                &[Value::Int(x)],
                "nl-compose-producer",
            )
            .map_err(|e| format!("producer '{producer_name}' failed to run on x={x}: {e}"))?;
            let a_int = match a_out {
                crate::runtime::Value::Int(v) => v,
                other => return Err(format!("producer returned non-int {other:?}")),
            };
            let b_expected = match &residual_code {
                None => a_int, // identity residual
                Some(hc) => {
                    let h_out = crate::runtime::execute_function(
                        hc,
                        &consumer_req.function_name,
                        &[Value::Int(a_int)],
                        "nl-compose-residual",
                    )
                    .map_err(|e| {
                        format!(
                            "residual '{}' failed to run on {a_int}: {e}",
                            consumer_req.function_name
                        )
                    })?;
                    match h_out {
                        crate::runtime::Value::Int(v) => v,
                        other => return Err(format!("residual returned non-int {other:?}")),
                    }
                }
            };
            composed.push(Example {
                inputs: vec![Value::Int(x)],
                expected: Value::Int(b_expected),
            });
        }
        // De-duplicate identical composed rows (a square op repeats x and -x);
        // keep distinct inputs so the search gets the widest signal.
        composed.dedup();

        // Build B's COMPOSED problem. The signature/category are inferred from the
        // composed examples (single i64 -> i64). Holdouts are empty (the composed
        // rows are themselves derived by an independent reference: the solved
        // producer+residual, NOT B's own search).
        let signature = infer_signature(consumer_name, &composed);
        let signature = Box::leak(signature.into_boxed_str());
        let category = Box::leak(consumer_req.category.clone().into_boxed_str());
        let description = Box::leak(consumer_req.description.clone().into_boxed_str());
        let problem = Problem {
            name: consumer_name.to_string(),
            category,
            description,
            signature,
            examples: composed,
            holdouts: Vec::new(),
            reference_code: "",
            synthetic_args: Vec::new(),
            synthetic_values: Vec::new(),
            recursive_allowed: false,
            tree_input: false,
            explicit_stack: false,
            functions: Vec::new(),
        };

        // Register the producer A as a real callable: name (for emission), arity,
        // its full Mog source (for the strict-verify prepend), and an `eval` that
        // RUNS A so a `Call(A, args)` candidate is verified end-to-end.
        let producer_src = producer_code.to_string();
        let producer_fn = producer_name.to_string();
        let callee = NamedCallable {
            name: producer_fn.clone(),
            n_args: 1,
            source: producer_src.clone(),
            eval: Box::new(move |xs: &[i64]| {
                if xs.len() != 1 {
                    return None;
                }
                match crate::runtime::execute_function(
                    &producer_src,
                    &producer_fn,
                    &[Value::Int(xs[0])],
                    "nl-compose-callee",
                ) {
                    Ok(crate::runtime::Value::Int(v)) => Some(v),
                    _ => None,
                }
            }),
        };

        let result = crate::enumerative::synthesize_scalar_with_callees(&problem, &[callee])
            .ok_or_else(|| {
                "Call-node search found no program calling the producer for the composed examples"
                    .to_string()
            })?;
        if !result.success {
            return Err(format!(
                "Call-node search did not verify (method={}, err={:?})",
                result.method, result.error
            ));
        }
        // The emitted code MUST genuinely CALL the producer (not an inlined
        // re-derivation). Refuse silently-inlined results so the accept-criterion
        // (a real call naming A) is never gamed.
        if !crate::agent::repo::body_calls_fn(&result.code, producer_name) {
            return Err(format!(
                "Call-node search produced a program that does NOT call '{producer_name}' \
                 (inlined): {}",
                result.code
            ));
        }
        Ok(result)
    }

    /// P2C scalar synthesis with a caller-chosen function name (LOOP-6 backend door).
    ///
    /// Classifies `description` as a linear `"X then Y"` compositional chain,
    /// manufactures examples via `problem_from_reference`, synthesizes, and
    /// strict-verifies — zero inline examples required.
    pub fn synthesize_p2c_scalar_named(
        &self,
        name: &str,
        description: &str,
    ) -> Result<crate::solver::SolveResult, String> {
        let registry = self.registry_clone().map_err(|e| e.to_string())?;
        match crate::reference_nl::classify_compositional(description, &registry) {
            crate::reference_nl::CompositionalIntake::Compositional { chain, .. } => {
                let signature = if chain.first().map(|s| s.arity).unwrap_or(1) == 2 {
                    format!("fn {name}(a: i64, b: i64) -> i64")
                } else {
                    format!("fn {name}(x: i64) -> i64")
                };
                self.solve_compositional_component(name, &signature, &chain)
            }
            crate::reference_nl::CompositionalIntake::Unresolvable(reason) => Err(format!(
                "P2C description for '{name}' has an unresolvable atom: {reason}"
            )),
            crate::reference_nl::CompositionalIntake::NotCompositional => Err(format!(
                "P2C description for '{name}' is not a compositional scalar chain: {description:?}"
            )),
        }
    }

    /// Unified prose scalar synthesis (LOOP-7) — tries doors until one succeeds:
    /// 1. compositional `then`-chain (P2C),
    /// 2. single registry unary op,
    /// 3. NL comprehend + strict-verify.
    ///
    /// Returns `(SolveResult, door_tag)` where `door_tag` is one of
    /// `"prose:p2c"`, `"prose:single-op"`, `"prose:project"`, `"prose:seeded"`, or `"prose:nl-desc"`.
    pub fn synthesize_prose_scalar_named(
        &self,
        name: &str,
        description: &str,
    ) -> Result<(crate::solver::SolveResult, &'static str), String> {
        let registry = self.registry_clone().map_err(|e| e.to_string())?;
        match crate::reference_nl::classify_compositional(description, &registry) {
            crate::reference_nl::CompositionalIntake::Compositional { chain, .. } => {
                let signature = if chain.first().map(|s| s.arity).unwrap_or(1) == 2 {
                    format!("fn {name}(a: i64, b: i64) -> i64")
                } else {
                    format!("fn {name}(x: i64) -> i64")
                };
                let res = self.solve_compositional_component(name, &signature, &chain)?;
                Ok((res, "prose:p2c"))
            }
            crate::reference_nl::CompositionalIntake::Unresolvable(reason) => Err(format!(
                "prose description for '{name}' has an unresolvable compositional atom: {reason}"
            )),
            crate::reference_nl::CompositionalIntake::NotCompositional => {
                if let Some(step) =
                    crate::reference_nl::resolve_best_scalar_op(description, &registry)
                {
                    if step.arity == 1 {
                        let signature = format!("fn {name}(x: i64) -> i64");
                        if let Ok(res) = self.solve_compositional_component(
                            name,
                            &signature,
                            std::slice::from_ref(&step),
                        ) {
                            return Ok((res, "prose:single-op"));
                        }
                    }
                }
                if let Ok(res) = self.synthesize_project_clause_named(name, description) {
                    return Ok((res, "prose:project"));
                }
                if let Ok(res) = self.synthesize_registry_seeded_clause_named(name, description) {
                    return Ok((res, "prose:seeded"));
                }
                let res = self.synthesize_from_description_strict(name, description)?;
                if !self.verify_mog_against_registry_examples(&res.code, name, description) {
                    return Err(format!(
                        "NL description for '{name}' synthesized code that disagrees with registry examples"
                    ));
                }
                Ok((res, "prose:nl-desc"))
            }
        }
    }

    /// Single-clause project synthesis (LOOP-8 affine/polynomial prose door).
    ///
    /// Routes `A function NAME that DESCRIPTION` through the real
    /// `synthesize_project` comprehend path — the same door inline-example
    /// backends use, but without requiring `name(x)=y` literals in the text.
    pub fn synthesize_project_clause_named(
        &self,
        name: &str,
        description: &str,
    ) -> Result<crate::solver::SolveResult, String> {
        let text = format!("A function {name} that {description}.");
        let (solved, skipped) = self.synthesize_project(&text)?;
        if !skipped.is_empty() {
            return Err(format!(
                "project clause for '{name}' skipped component(s): {skipped:?}"
            ));
        }
        if solved.len() == 1 && solved[0].1.success {
            let res = solved.into_iter().next().unwrap().1;
            if Self::validates_project_scalar_result(&res.code, name) {
                return Ok(res);
            }
            return Err(format!(
                "project clause for '{name}' synthesized non-scalar or invalid i64 behaviour"
            ));
        }
        let res = solved
            .into_iter()
            .find(|(n, r)| n == name && r.success)
            .map(|(_, r)| r)
            .ok_or_else(|| {
                format!("project clause for '{name}' did not yield a successful synthesis")
            })?;
        if !Self::validates_project_scalar_result(&res.code, name) {
            return Err(format!(
                "project clause for '{name}' synthesized non-scalar or invalid i64 behaviour"
            ));
        }
        Ok(res)
    }

    fn validates_project_scalar_result(mog: &str, name: &str) -> bool {
        let header = mog.lines().next().unwrap_or("").to_lowercase();
        if !header.contains("-> i64") {
            return false;
        }
        if header.contains('[') || header.contains("vec<") {
            return false;
        }
        for x in [0_i64, 1, 3] {
            if let Ok(crate::runtime::Value::Int(_)) =
                crate::runtime::execute_function(mog, name, &[crate::benchmark::Value::Int(x)], name)
            {
                return true;
            }
        }
        false
    }

    /// Registry-example-seeded project clause (LOOP-8).
    ///
    /// When comprehend yields registry `example_cases`, format them as inline
    /// `name(x)=y` literals and re-enter the project door — no hand-grader.
    pub fn synthesize_registry_seeded_clause_named(
        &self,
        name: &str,
        description: &str,
    ) -> Result<crate::solver::SolveResult, String> {
        use linguigenesis_core::coding_requirements::LiteralValue;
        let input = format!("A function {name} that {description}.");
        let req = match self.nl_to_requirement(&input) {
            Ok(req) => req,
            Err(BridgeError::ClarificationNeeded { partial, .. }) if partial.examples.len() >= 2 => {
                partial
            }
            Err(e) => return Err(e.to_string()),
        };
        let mut literals = Vec::new();
        for ex in req.examples.iter().take(6) {
            if ex.inputs.len() == 1 {
                if let (LiteralValue::Int(x), LiteralValue::Int(y)) = (&ex.inputs[0], &ex.expected)
                {
                    literals.push(format!("{name}({x})={y}"));
                }
            }
        }
        if literals.len() < 2 {
            return Err(format!(
                "registry seed for '{name}' has fewer than 2 formattable i64 examples"
            ));
        }
        let text = format!(
            "A function {name} that {description}, {}.",
            literals.join(" and ")
        );
        let (solved, skipped) = self.synthesize_project(&text)?;
        if !skipped.is_empty() {
            return Err(format!(
                "registry-seeded project for '{name}' skipped: {skipped:?}"
            ));
        }
        if solved.len() == 1 && solved[0].1.success {
            let res = solved.into_iter().next().unwrap().1;
            if Self::validates_project_scalar_result(&res.code, name) {
                return Ok(res);
            }
            return Err(format!(
                "registry-seeded project for '{name}' produced invalid scalar i64 behaviour"
            ));
        }
        let res = solved
            .into_iter()
            .find(|(n, r)| n == name && r.success)
            .map(|(_, r)| r)
            .ok_or_else(|| {
                format!("registry-seeded project for '{name}' did not synthesize")
            })?;
        if !Self::validates_project_scalar_result(&res.code, name) {
            return Err(format!(
                "registry-seeded project for '{name}' produced invalid scalar i64 behaviour"
            ));
        }
        Ok(res)
    }

    /// When registry `example_cases` exist for a clause, Mog execution must agree.
    fn verify_mog_against_registry_examples(
        &self,
        mog: &str,
        name: &str,
        description: &str,
    ) -> bool {
        use linguigenesis_core::coding_requirements::LiteralValue;
        let input = format!("A function {name} that {description}.");
        let req = match self.nl_to_requirement(&input) {
            Ok(req) => req,
            Err(BridgeError::ClarificationNeeded { partial, .. }) if !partial.examples.is_empty() => {
                partial
            }
            Err(_) => return true,
        };
        if req.examples.is_empty() {
            return true;
        }
        let mut checked = 0;
        for ex in req.examples.iter().take(6) {
            if ex.inputs.len() != 1 {
                continue;
            }
            let (LiteralValue::Int(x), LiteralValue::Int(y)) = (&ex.inputs[0], &ex.expected) else {
                continue;
            };
            checked += 1;
            let got = match crate::runtime::execute_function(
                mog,
                name,
                &[crate::benchmark::Value::Int(*x)],
                name,
            ) {
                Ok(crate::runtime::Value::Int(n)) => n,
                _ => return false,
            };
            if got != *y {
                return false;
            }
        }
        checked > 0
    }

    /// NL description → solve → strict-verify (LOOP-7 third prose door).
    fn synthesize_from_description_strict(
        &self,
        name: &str,
        description: &str,
    ) -> Result<crate::solver::SolveResult, String> {
        if let Some(outcome) = self.try_compose_pipeline(description) {
            let res = outcome?;
            let solved = res.into_solve_result();
            if !solved.success {
                return Err(format!(
                    "pipeline synthesis for '{name}' failed: {:?}",
                    solved.error
                ));
            }
            return Ok(solved);
        }
        let req = match self.nl_to_requirement(description) {
            Ok(req) => req,
            Err(BridgeError::ClarificationNeeded { questions, .. }) => {
                return Err(format_clarification_prompt(&questions));
            }
            Err(e) => return Err(e.to_string()),
        };
        let problem = self
            .problem_from_requirement(&req, Some(name))
            .map_err(|e| e.to_string())?;
        let solved = crate::solver::solve_problem(&problem);
        if !solved.success {
            return Err(format!(
                "NL description for '{name}' did not synthesize (method={}, err={:?})",
                solved.method, solved.error
            ));
        }
        crate::runtime::verify_problem_code_strict(&problem, &solved.code).map_err(|e| {
            format!(
                "NL description for '{name}' OVERFIT — strict holdout verification failed: {e}"
            )
        })?;
        Ok(solved)
    }

    /// P2C-AUTO-CONTRACT a single COMPOSITIONAL component (a described scalar
    /// `"X then Y"` chain) ALL THE WAY into a verified standalone function — the
    /// same path `reference_nl`'s `drive_end_to_end` uses, factored so
    /// [`Self::synthesize_project`] can route each component clause through it.
    ///
    /// Reuses (NO new synthesizer): [`Self::emit_scalar_reference`] (emit the
    /// independent composed reference), [`crate::benchmark::problem_from_reference`]
    /// (RUN that reference to manufacture seed examples + the holdout oracle — zero
    /// human examples), [`crate::solver::solve_problem`] (synthesize the composed
    /// fn from those examples), and [`crate::runtime::verify_problem_code_strict`]
    /// (strict-verify the solved fn on FRESH reference-labelled holdouts so an
    /// overfit is refused, not written). The returned `SolveResult.code` is a
    /// self-contained `fn <name>` whose name matches `name` — ready for the
    /// existing multi-file writer.
    fn solve_compositional_component(
        &self,
        name: &str,
        signature: &str,
        chain: &[crate::reference_nl::CompositionalStep],
    ) -> Result<crate::solver::SolveResult, String> {
        let reference = self.emit_scalar_reference(name, chain)?;
        let sig_static: &'static str = Box::leak(signature.to_string().into_boxed_str());
        let ref_static: &'static str = Box::leak(reference.into_boxed_str());
        let mut problem = crate::benchmark::problem_from_reference(name, sig_static, ref_static)
            .map_err(|e| format!("compositional reference unrunnable: {e}"))?;
        problem.category = Box::leak("nl-compose-scalar".to_string().into_boxed_str());
        let solved = crate::solver::solve_problem(&problem);
        if !solved.success {
            return Err(format!(
                "solver could not synthesize compositional '{name}' (method={}, err={:?})",
                solved.method, solved.error
            ));
        }
        crate::runtime::verify_problem_code_strict(&problem, &solved.code).map_err(|e| {
            format!(
                "compositional '{name}' OVERFIT — strict holdout verification failed: {e}\nCODE:\n{}",
                solved.code
            )
        })?;
        Ok(solved)
    }

    /// Synthesize a single-arg (`i64 -> i64`) op named `op_fn` from its registry
    /// entity through the same single-op door producers use. Errors if the op is
    /// absent, not synthesizable, not single-arg, or fails to solve. Used to
    /// derive a consumer's RESIDUAL h.
    fn synthesize_named_unary_op(&self, op_fn: &str) -> Result<String, String> {
        use linguigenesis_core::entity::EntityType;
        let registry = self.registry_clone().map_err(|e| e.to_string())?;
        let entity = registry
            .get_by_type(&EntityType::Function)
            .into_iter()
            .find(|e| {
                e.get_property("default_fn_name").map(|f| f == op_fn).unwrap_or(false)
                    || e.lemma == op_fn
            })
            .ok_or_else(|| format!("op '{op_fn}' not in registry"))?;
        let req = SynthesisRequirement::from_operation_entity(&entity)
            .ok_or_else(|| format!("op '{op_fn}' is not synthesizable (no example_cases)"))?;
        // Must be a single-arg scalar op for the unary-residual composition.
        let arity = req.examples.first().map(|e| e.inputs.len()).unwrap_or(0);
        if arity != 1 {
            return Err(format!("op '{op_fn}' is not single-arg (arity {arity})"));
        }
        let result = self
            .synthesize_from_requirement(&req, Some(op_fn))
            .map_err(|e| format!("residual '{op_fn}' synthesis: {e}"))?;
        if !result.success {
            return Err(format!(
                "residual '{op_fn}' did not solve (method={}, err={:?})",
                result.method, result.error
            ));
        }
        Ok(result.code)
    }

    /// Synthesize a scalar-`i64` op named `op_fn` (ANY arity) from its registry
    /// entity through the SAME single-op door producers use
    /// ([`Self::synthesize_primitive`]'s path, arity-agnostic). Returns the
    /// verified Mog source whose `fn` is named `op_fn`. Used by
    /// [`Self::emit_scalar_reference`] to obtain each chain primitive's emittable
    /// body. Errors if the op is absent, not synthesizable, or fails to solve.
    fn synthesize_named_scalar_op(&self, op_fn: &str) -> Result<String, String> {
        use linguigenesis_core::entity::EntityType;
        let registry = self.registry_clone().map_err(|e| e.to_string())?;
        let entity = registry
            .get_by_type(&EntityType::Function)
            .into_iter()
            .find(|e| {
                e.get_property("default_fn_name").map(|f| f == op_fn).unwrap_or(false)
                    || e.lemma == op_fn
            })
            .ok_or_else(|| format!("op '{op_fn}' not in registry"))?;
        let req = SynthesisRequirement::from_operation_entity(&entity)
            .ok_or_else(|| format!("op '{op_fn}' is not synthesizable (no example_cases)"))?;
        let result = self
            .synthesize_from_requirement(&req, Some(op_fn))
            .map_err(|e| format!("primitive '{op_fn}' synthesis: {e}"))?;
        if !result.success {
            return Err(format!(
                "primitive '{op_fn}' did not solve (method={}, err={:?})",
                result.method, result.error
            ));
        }
        Ok(result.code)
    }

    /// P2C-PROMPT-TO-CONTRACT: emit an INDEPENDENT runnable reference for a linear
    /// scalar composition (generalises [`emit_pipeline_reference`] beyond
    /// map/fold). Each chain primitive's verified single-op body is emitted once
    /// (deduped), then the composed entry fn NESTS them in DESCRIBED order around
    /// the head's params: `head(a, b)` (or `head(x)`) threaded through each unary
    /// tail, e.g. `max` then `triple` → `return triple(max(a, b));`. The emitted
    /// source is fed UNCHANGED to [`crate::benchmark::problem_from_reference`],
    /// which RUNS it to manufacture the seed examples + holdout oracle (zero human
    /// examples). Returns the full Mog source.
    ///
    /// `chain[0]` is the head (arity 1 or 2, sets the params); every later step
    /// must be arity 1 (threaded onto the running scalar) — the caller
    /// ([`crate::reference_nl::classify_compositional`]) already enforces this.
    pub fn emit_scalar_reference(
        &self,
        name: &str,
        chain: &[crate::reference_nl::CompositionalStep],
    ) -> Result<String, String> {
        if chain.is_empty() {
            return Err("empty composition chain".to_string());
        }
        let head_arity = chain[0].arity;
        if head_arity != 1 && head_arity != 2 {
            return Err(format!("unsupported head arity {head_arity}"));
        }

        // Synthesize + emit each DISTINCT primitive body once (the chain may repeat
        // an op, but its body only needs to appear a single time).
        let mut out = String::new();
        let mut emitted: Vec<String> = Vec::new();
        for step in chain {
            if emitted.contains(&step.fn_name) {
                continue;
            }
            if step != &chain[0] && step.arity != 1 {
                return Err(format!(
                    "non-head step '{}' has arity {} (must be unary)",
                    step.fn_name, step.arity
                ));
            }
            let body = self.synthesize_named_scalar_op(&step.fn_name)?;
            emitted.push(step.fn_name.clone());
            out.push_str(body.trim_end());
            out.push_str("\n\n");
        }

        // Nest the chain head→...→tail around the head's params.
        let (params, mut expr) = if head_arity == 2 {
            ("a: i64, b: i64", format!("{}(a, b)", chain[0].fn_name))
        } else {
            ("x: i64", format!("{}(x)", chain[0].fn_name))
        };
        for step in &chain[1..] {
            expr = format!("{}({})", step.fn_name, expr);
        }
        out.push_str(&format!(
            "fn {name}({params}) -> i64 {{\n    return {expr};\n}}\n"
        ));
        Ok(out)
    }

    /// Synthesize a registry op identified BY LEMMA, emitting it under `as_name`.
    /// Lemma (not `default_fn_name`) is required because several distinct string
    /// ops share the same `default_fn_name` (`transform`); they are told apart —
    /// and synthesized from their OWN `example_cases` — only by lemma. Reuses the
    /// existing solver via `from_operation_entity` + `synthesize_from_requirement`
    /// (NO new synthesizer).
    fn synthesize_op_by_lemma(&self, lemma: &str, as_name: &str) -> Result<String, String> {
        let registry = self.registry_clone().map_err(|e| e.to_string())?;
        let entity = registry
            .get_by_lemma(lemma)
            .ok_or_else(|| format!("op lemma '{lemma}' not in registry"))?;
        let req = SynthesisRequirement::from_operation_entity(&entity)
            .ok_or_else(|| format!("op '{lemma}' is not synthesizable (no example_cases)"))?;
        let result = self
            .synthesize_from_requirement(&req, Some(as_name))
            .map_err(|e| format!("primitive '{lemma}' synthesis: {e}"))?;
        if !result.success {
            return Err(format!(
                "primitive '{lemma}' did not solve (method={}, err={:?})",
                result.method, result.error
            ));
        }
        Ok(result.code)
    }

    /// P2C WIDEN (BUILD-A): emit an INDEPENDENT runnable reference for an ARRAY
    /// composition — ordered whole-array `[i64]→[i64]` map transforms, optionally
    /// terminated by a reduce. Each distinct map op's verified body is emitted
    /// once, the running array is threaded through them via fresh bindings
    /// (`m0 = map0(arr); m1 = map1(m0); ...`), then either the optional fold is
    /// emitted over the last array (scalar `[i64]→i64` output, reusing
    /// [`emit_fold_over_named_array`] / the same fold classifiers the pipeline
    /// door uses) or the last array is returned (`[i64]→[i64]`). Fed UNCHANGED to
    /// [`crate::benchmark::problem_from_reference`].
    pub fn emit_array_reference(
        &self,
        name: &str,
        maps: &[crate::reference_nl::DomainStep],
        reduce: Option<&crate::reference_nl::DomainStep>,
    ) -> Result<String, String> {
        if maps.is_empty() {
            return Err("empty array composition (no map head)".to_string());
        }
        let mut out = String::new();
        // Emit each DISTINCT map body once (keyed by fn_name).
        let mut emitted: Vec<String> = Vec::new();
        for m in maps {
            if emitted.contains(&m.fn_name) {
                continue;
            }
            let body = self.synthesize_op_by_lemma(&m.lemma, &m.fn_name)?;
            emitted.push(m.fn_name.clone());
            out.push_str(body.trim_end());
            out.push_str("\n\n");
        }

        // Thread the running array through the map chain via fresh bindings.
        let mut body = String::new();
        let mut cur = "arr".to_string();
        for (i, m) in maps.iter().enumerate() {
            let var = format!("m{i}");
            body.push_str(&format!("    {var}: [i64] = {}({cur});\n", m.fn_name));
            cur = var;
        }

        match reduce {
            Some(r) => {
                // Classify the fold by EXECUTING the synthesized reduce op (never
                // name-keyed) — an array reduce ([i64]→i64) is probed with arrays,
                // a binary fold seed (i64,i64→i64, e.g. `add` for "sum") with
                // scalar pairs. Reuses the pipeline door's classifiers.
                let reduce_code = self.synthesize_op_by_lemma(&r.lemma, &r.fn_name)?;
                let fold = if r.input_types == "i64,i64" {
                    classify_binary_fold(&reduce_code, &r.fn_name)
                } else {
                    classify_array_fold(&reduce_code, &r.fn_name)
                }
                .ok_or_else(|| {
                    format!("could not classify the reduce fold for op '{}'", r.fn_name)
                })?;
                let fold_body = emit_fold_over_named_array(fold, &cur);
                out.push_str(&format!(
                    "fn {name}(arr: [i64]) -> i64 {{\n{body}{fold_body}}}\n"
                ));
            }
            None => {
                out.push_str(&format!(
                    "fn {name}(arr: [i64]) -> [i64] {{\n{body}    return {cur};\n}}\n"
                ));
            }
        }
        Ok(out)
    }

    /// P2C WIDEN (BUILD-A): emit an INDEPENDENT runnable reference for a STRING
    /// composition — ordered `string→string` transforms nested inner→outer
    /// around the input (`step_n(...step_0(s)...)`). Each step's verified body is
    /// synthesized from ITS OWN entity (by lemma, since several string ops share
    /// `default_fn_name`) and emitted under a UNIQUE helper name (its lemma).
    /// Fed UNCHANGED to [`crate::benchmark::problem_from_reference`].
    pub fn emit_string_reference(
        &self,
        name: &str,
        steps: &[crate::reference_nl::DomainStep],
    ) -> Result<String, String> {
        if steps.is_empty() {
            return Err("empty string composition".to_string());
        }
        let mut out = String::new();
        let mut emitted: Vec<String> = Vec::new();
        for s in steps {
            // Helper name = lemma (unique per op, unlike the shared fn_name).
            if emitted.contains(&s.lemma) {
                continue;
            }
            let body = self.synthesize_op_by_lemma(&s.lemma, &s.lemma)?;
            emitted.push(s.lemma.clone());
            out.push_str(body.trim_end());
            out.push_str("\n\n");
        }
        // Nest in described order: steps[0] is innermost (applied first).
        let mut expr = "s".to_string();
        for s in steps {
            expr = format!("{}({})", s.lemma, expr);
        }
        out.push_str(&format!(
            "fn {name}(s: string) -> string {{\n    return {expr};\n}}\n"
        ));
        Ok(out)
    }

    /// TEST-SUPPORT: (lemma, default_fn_name) for every example-bearing op in the
    /// merged registry. Lets a harness build data-driven derivational paraphrases
    /// from the real op vocabulary instead of a hand list.
    pub fn op_lemmas(&self) -> Vec<(String, String)> {
        use linguigenesis_core::entity::EntityType;
        let registry = match self.registry_clone() {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        let mut out = Vec::new();
        for e in registry.get_by_type(&EntityType::Function) {
            if e.get_property("example_cases").is_some() {
                let fnn = e
                    .get_property("default_fn_name")
                    .cloned()
                    .unwrap_or_else(|| e.lemma.clone());
                out.push((e.lemma.clone(), fnn));
            }
        }
        out
    }

    /// TEST-SUPPORT: ranked op candidates `(fn_name, score, method)` for a surface,
    /// highest first, via the SAME resolver the gate uses. The `method` string
    /// separates emergent-lens contributions (frame/prime/root/phonestheme) from
    /// the curated scorers, so a harness can attribute and measure recall lift.
    pub fn probe_op_candidates(&self, word: &str) -> Vec<(String, f32, String)> {
        let registry = match self.registry_clone() {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        let resolver = EntityResolver::new(registry);
        resolver
            .rank_candidates(word)
            .into_iter()
            .filter(|c| linguigenesis_core::entity_resolution::is_operation(&c.entity))
            .map(|c| {
                let fnn = c
                    .entity
                    .get_property("default_fn_name")
                    .cloned()
                    .unwrap_or_else(|| c.entity.lemma.clone());
                (fnn, c.evidence.score, c.evidence.method.to_string())
            })
            .collect()
    }

    /// TEST-SUPPORT: top op resolution for a surface as `(fn_name, score, method)`,
    /// following synonym/similar edges to the canonical example-bearing op (as
    /// `resolve_operation_surface` does), so the fn-name is the real synthesis
    /// target. The `method` distinguishes an emergent lens from a curated scorer.
    pub fn probe_resolution(&self, word: &str) -> Option<(String, f32, String)> {
        let registry = self.registry_clone().ok()?;
        let resolver = EntityResolver::new(registry);
        let r = resolver.resolve_operation_surface(word)?;
        let fnn = r
            .entity
            .get_property("default_fn_name")
            .cloned()
            .unwrap_or_else(|| r.entity.lemma.clone());
        Some((fnn, r.evidence.score, r.evidence.method.to_string()))
    }

    /// PHRASE-level op resolution over a whole request: matches a MULTI-WORD op
    /// reference in the prose against each synthesizable op's own lemma tokens
    /// ("reverse the string" → `reverse_string`) and any data `phrase_surfaces`,
    /// in order, each word by the emergent morphology tiers. The resolution-side
    /// half of the NL-vocabulary lane (the universal agent's surface-derivation
    /// miner is the emission half — see MASTER_ROADMAP 0.0591). Returns
    /// `(default_fn_name, score)`.
    pub fn resolve_phrase_op(&self, text: &str) -> Option<(String, f32)> {
        use linguigenesis_core::entity_resolution::resolve_phrase_operation;
        let registry = self.registry_clone().ok()?;
        let resolver = EntityResolver::new(registry);
        let tokens: Vec<String> = text
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|t| !t.is_empty())
            .map(str::to_string)
            .collect();
        let (r, _span) = resolve_phrase_operation(&resolver, &tokens)?;
        let fnn = r
            .entity
            .get_property("default_fn_name")
            .cloned()
            .unwrap_or_else(|| r.entity.lemma.clone());
        Some((fnn, r.evidence.score))
    }

    /// TEST-SUPPORT: resolve a single surface word to its highest-confidence
    /// programming op via the SAME emergent resolver the gate uses
    /// ([`resolved_content_ops`]). Returns `(default_fn_name, score)` for the
    /// top op, or `None` if the word resolves to no Function/Operator op.
    /// Used by the WordNet-recall benchmark's resolution/lift/refuse probes.
    pub fn resolve_op_probe(&self, word: &str) -> Option<(String, f32)> {
        use linguigenesis_core::entity::EntityType;
        let registry = self.registry_clone().ok()?;
        let resolver = EntityResolver::new(registry);
        let resolved = resolver.resolve_operation_surface(word)?;
        if !matches!(
            resolved.entity.entity_type,
            EntityType::Function | EntityType::Operator
        ) {
            return None;
        }
        let fn_name = resolved
            .entity
            .get_property("default_fn_name")
            .cloned()
            .unwrap_or_else(|| resolved.entity.lemma.clone());
        Some((fn_name, resolved.evidence.score))
    }

    /// TEST-SUPPORT: every DISTINCT op (`default_fn_name`) a word resolves to at
    /// or above `floor`, across ALL ranked candidates — used to assert
    /// cluster-separation (a paraphrase must not resolve to two different ops).
    pub fn resolve_ops_above_floor(&self, word: &str, floor: f32) -> Vec<String> {
        use linguigenesis_core::entity::EntityType;
        let Ok(registry) = self.registry_clone() else {
            return Vec::new();
        };
        let resolver = EntityResolver::new(registry.clone());
        let mut ops: Vec<String> = Vec::new();
        for cand in resolver.rank_candidates(word) {
            if cand.evidence.score < floor {
                continue;
            }
            // Map the candidate to its canonical op (follow synonym to the
            // example-bearing seed), exactly as the resolver's op path does.
            let Some(op) = resolver.resolve_operation_surface(&cand.entity.lemma) else {
                continue;
            };
            if !matches!(
                op.entity.entity_type,
                EntityType::Function | EntityType::Operator
            ) {
                continue;
            }
            let fn_name = op
                .entity
                .get_property("default_fn_name")
                .cloned()
                .unwrap_or_else(|| op.entity.lemma.clone());
            if !ops.contains(&fn_name) {
                ops.push(fn_name);
            }
        }
        ops
    }

    /// TEST-SUPPORT: prove a paraphrase RESOLVES to `expected_op` via the WordNet
    /// edge AND that op synthesizes a program passing strict differential
    /// verification on FRESH holdouts the verifier samples and labels by RUNNING
    /// an INDEPENDENT reference (`problem_from_reference` path — never
    /// example_cases). Errors carry the failure reason.
    pub fn resolve_and_strict_verify(
        &self,
        paraphrase: &str,
        expected_op: &str,
        ref_name: &'static str,
        ref_signature: &'static str,
        reference_code: &'static str,
    ) -> Result<(), String> {
        // 1. RESOLUTION: the paraphrase must reach `expected_op` at the floor.
        match self.resolve_op_probe(paraphrase) {
            Some((fn_name, score)) if fn_name == expected_op && score >= OP_RESOLVE_FLOOR => {}
            Some((fn_name, score)) => {
                return Err(format!(
                    "{paraphrase:?} resolved to {fn_name:?}@{score:.3}, expected {expected_op:?}@>={OP_RESOLVE_FLOOR}"
                ));
            }
            None => return Err(format!("{paraphrase:?} did not resolve to any op")),
        }
        // 2. SYNTHESIS + STRICT FRESH-HOLDOUT VERIFY via the reference path.
        let mut problem = crate::benchmark::problem_from_reference(
            ref_name,
            ref_signature,
            reference_code,
        )
        .map_err(|e| format!("reference unrunnable: {e}"))?;
        problem.category = "nl-wordnet-recall";
        let solved = crate::solver::solve_problem(&problem);
        if !solved.success {
            return Err(format!(
                "resolved to {expected_op} but solver could not synthesize (method={}, err={:?})",
                solved.method, solved.error
            ));
        }
        crate::runtime::verify_problem_code_strict(&problem, &solved.code).map_err(|e| {
            format!("OVERFIT — strict holdout verification failed: {e}\nCODE:\n{}", solved.code)
        })
    }

    /// Get belief state from NL (for debugging)
    pub fn get_belief_state(&self, input: &str) -> Result<BeliefState, BridgeError> {
        let mut comp = self
            .comprehension
            .write()
            .map_err(|_| BridgeError::LockError)?;

        Ok(comp.parse(input))
    }

    /// Ask knowledge query
    pub fn query_knowledge(&self, entity_lemma: &str) -> Option<String> {
        let registry = self.registry.read().ok()?;
        if let Some(entity) = registry.get_by_lemma(entity_lemma) {
            if !entity.definitions.is_empty() {
                return Some(entity.definitions.join("; "));
            }
        }
        None
    }
}

impl Default for LinguigenesisBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Bridge errors
#[derive(Debug, thiserror::Error)]
pub enum BridgeError {
    #[error("Lock error")]
    LockError,

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("No examples generated")]
    NoExamples,

    #[error("Clarification needed")]
    ClarificationNeeded {
        partial: SynthesisRequirement,
        questions: Vec<ClarificationQuestion>,
    },

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

/// Public belief state representation for analysis
#[derive(Debug, Clone)]
pub struct BridgeBeliefState {
    pub intent_type: String,
    pub entities: Vec<String>,
    pub confidence: f64,
}

/// Structural soundness gate for a *registry-derived* confident match.
///
/// Returns `Some(reason)` (a marker pushed into `unresolved`, NOT a user-facing
/// blocklist string) when the requirement should be DOWNGRADED to
/// ClarificationNeeded instead of declared solved. Every signal is computed from
/// the parse the bridge already holds — never a hardcoded bad-phrase list:
///
///   1. **Discrimination floor** — registry NL matches carry no reference impl,
///      so the only generalization evidence is the canned `example_cases`. Fewer
///      than two examples cannot discriminate the intended function from
///      alternatives (e.g. one (`[1,2,3]->[3,2,1]`) pair is consistent with
///      reverse, rotate, sort-desc, ...). Require >=2.
///   2. **Request/signature TYPE mismatch** — the request names a value type
///      (string / array / list) absent from the resolved op's signature
///      (e.g. "reverse a STRING" resolving to `fn reverse(a: Vec<i64>)`).
///   3. **Operation not named by the request** — none of the request's content
///      words corresponds to the resolved op's identity (function name /
///      description / category). The op was guessed from leftover tokens while
///      the actual content words ("parse", "csv", "file") were dropped.
///
/// Inline-example requests (the user supplied their own I/O) are exempt: their
/// evidence is the demonstrated behaviour, not a registry guess. We detect that
/// the same way the comprehension layer does — a non-generic function name with
/// the user's own examples is treated as user-specified.
/// Why the fail-closed gate declined to synthesize confidently. The category lets
/// callers choose WHICH refusals are hard (a genuine confidently-wrong resolution
/// that must REFUSE everywhere) vs the soft "this is a composition the single-op
/// solver may still partially handle" case, which the agent path lets fall through
/// to the solver rather than refusing outright.
#[derive(Clone, Debug, PartialEq, Eq)]
enum GateCategory {
    /// Out-of-domain / type / unresolved / thin-spec: a confidently-WRONG single-op
    /// resolution. Must refuse on every path.
    Hard,
    /// The request names >1 distinct operation (a composition) that the pipeline
    /// path did not build. Soft: the agent's single-op solver may still synthesize
    /// the primary op, so the agent does not refuse on this alone.
    CompositionUnsupported,
}

/// Gate result: the refusal category plus its human-readable reason.
#[derive(Clone, Debug)]
struct GateRefusal {
    category: GateCategory,
    reason: String,
}

/// Back-compat `Option<String>` view used by `nl_to_requirement` (refuses on ANY
/// category — the comprehension front door fails closed on every gate signal).
fn unsound_confident_solve(
    input: &str,
    req: &SynthesisRequirement,
    registry: &Registry,
) -> Option<String> {
    unsound_confident_solve_categorized(input, req, registry).map(|r| r.reason)
}

fn unsound_confident_solve_categorized(
    input: &str,
    req: &SynthesisRequirement,
    registry: &Registry,
) -> Option<GateRefusal> {
    use linguigenesis_core::nl_tokens::tokenize_lower;

    // Inline-example requests carry the user's OWN demonstrated I/O pairs as the
    // spec (registry comprehension is bypassed by `apply_inline_examples`), so
    // there is no registry-misresolution to guard against — exempt them. Detected
    // structurally with the SAME parser the comprehension uses, not a phrase
    // heuristic. (The separate "is a single-canned-example registry op actually
    // proven against FRESH holdouts?" concern is P2's job, deliberately NOT folded
    // in here — this gate only catches confident-WRONG resolution.)
    if !linguigenesis_core::inline_examples::parse_inline_examples(input).is_empty() {
        return None;
    }

    // STATEFUL RE-TARGET EXEMPTION (UNWALL-1-STATEFUL-NL): a requirement whose
    // examples carry the two-input per-tick shape `(state: i64, arr: [i64]) -> i64`
    // was BEHAVIORALLY re-targeted to a stateful reducer (`retarget_stateful_
    // reducer`) — the reduce op the request also named is SUBSUMED by the stateful
    // op (it IS the same reduction, threaded through a per-tick state), not a
    // dropped composition. The operation-identity gate below would otherwise read
    // the array-reduce word ("total"/"sum") as a dropped op and refuse. Detected
    // STRUCTURALLY from the example shape (Int seed + Array, Int result), not a
    // category-name or phrase test; a plain reduce can never carry this shape.
    let is_stateful_pertick_shape = !req.examples.is_empty()
        && req.examples.iter().all(|ex| {
            ex.inputs.len() == 2
                && matches!(ex.inputs[0], LiteralValue::Int(_))
                && matches!(ex.inputs[1], LiteralValue::Array(_))
                && matches!(ex.expected, LiteralValue::Int(_))
        });
    if is_stateful_pertick_shape {
        return None;
    }

    // EMERGENT operation inventory: ask the RESOLVER what each surface token names,
    // and let the resolved ENTITY itself decide operand-vs-operation. There is no
    // VALUE_NOUNS wordlist, no type-noun→signature table, and no `evidence.method`
    // string switch. A token is an *operand* (number/value/array/the/of/...) exactly
    // when it does not resolve to a high-confidence programming operation; that is
    // observed, never enumerated. (`resolved_content_ops` filters by
    // `entity_type ∈ {Function, Operator}` and `evidence.score >= OP_RESOLVE_FLOOR`.)
    let ops = resolved_content_ops(input, registry);

    // If linguigenesis-core COMPREHENDED the request as a transform+aggregate (or
    // reduce-only / map-only degenerate) array pipeline, it populated
    // `req.pipeline` — the request is *comprehensible*, not unsound. Composition is
    // built downstream (`try_compose_pipeline`); here we simply decline to
    // fail-closed so the pipeline path can run. Single-op requests (`pipeline =
    // None`) fall through unchanged.
    if req.pipeline.is_some() {
        return None;
    }

    // (1a) >=2-DISTINCT-EXAMPLE DISCRIMINATION FLOOR (HARDEN-2 FIX B), scoped to
    // SCALAR ops. A single-op request (pipeline = None, not inline-example — both
    // exempted above) whose op is SCALAR is verified examples-only by
    // `verify_problem_code_strict` (its `problem_from_requirement` problem has no
    // runnable reference, so `generated_holdouts` degrades to the seed rows). With
    // a single distinct row ANY program reproducing that one pair "passes",
    // letting a thin 1-row scalar spec overfit confidently-wrong. We therefore
    // require >=2 DISTINCT (input → expected) rows: enough to (a) reserve one as a
    // held-out generalization probe in `problem_from_requirement` and (b)
    // discriminate against trivial constant/identity overfits. Below the floor we
    // DOWNGRADE to clarification rather than emit confident code.
    //
    // ARRAY-domain ops (sort/reverse — any example carries a `LiteralValue::Array`
    // operand or result) are EXEMPT: they are synthesized + verified through the
    // array-transform path (`classify_array_transform_by_spec/_by_exec`), which
    // checks the candidate against the op's labelled (array → array) pairs — a
    // differential oracle, not examples-only — so a single registry row does not
    // overfit. Array-ness is read from the op's own example shapes, not a name/
    // type table.
    let op_is_array = req.examples.iter().any(|ex| {
        use linguigenesis_core::coding_requirements::LiteralValue;
        matches!(ex.expected, LiteralValue::Array(_))
            || ex.inputs.iter().any(|i| matches!(i, LiteralValue::Array(_)))
    });
    if !op_is_array {
        let distinct_examples = {
            let mut seen: Vec<&linguigenesis_core::coding_requirements::ExampleSpec> = Vec::new();
            for ex in &req.examples {
                if !seen.iter().any(|s| **s == *ex) {
                    seen.push(ex);
                }
            }
            seen.len()
        };
        if distinct_examples < 2 {
            return Some(GateRefusal {
                category: GateCategory::Hard,
                reason: format!(
                    "insufficient evidence to synthesize confidently: resolved scalar op '{}' carries \
                     only {} distinct example row(s) (need >=2 to verify generalization, not overfit a \
                     single pair); supply an explicit example or disambiguate",
                    req.function_name, distinct_examples
                ),
            });
        }
    }

    let sig_lower = req.signature.to_lowercase();

    // (1b) ARRAY-DOMAIN vs SCALAR-OP mismatch, derived emergently from SIGNATURES.
    // If the resolved requirement op is SCALAR (its signature carries no array
    // type) yet some request token resolves to an ARRAY-domain operation (an op
    // whose declared `input_types` contains a vector type), the request is about
    // arrays but was collapsed onto a scalar op — a confidently-wrong resolution
    // (e.g. "sum of squares OF AN ARRAY" → `fn add(i64,i64)`, where the array
    // operation word was dropped). This compares the operand domain the request
    // names against the resolved op's domain; it is NOT a value-noun wordlist.
    // (A genuinely array-typed req op — array_sum, reverse — carries an array type
    // in its own signature, so this never fires for them.)
    // `string`/`str`/`&str` are Collection-domain in the type lattice (a
    // string→string op is a Collection→Collection transform, exactly like
    // vec→vec), so a string signature is ARRAY-domain, not scalar — otherwise the
    // guard below wrongly rejects string ops (e.g. lowercase: fn transform(string)
    // -> string) as scalar-vs-array mismatches.
    let req_sig_is_array =
        sig_lower.contains('[') || sig_lower.contains("vec<") || sig_lower.contains("str");
    if !req_sig_is_array {
        if let Some(arr_word) = array_domain_word(input, registry, &req.function_name, &req.signature) {
            return Some(GateRefusal {
                category: GateCategory::Hard,
                reason: format!(
                    "no operation confidently resolved: request names an array operand ('{}') but \
                     resolved op '{}' has scalar signature '{}' (domain mismatch)",
                    arr_word, req.function_name, req.signature
                ),
            });
        }
    }

    // (2) Request value-type vs resolved-signature value-type MISMATCH, derived
    // emergently: a token that resolves (high-confidence) to a registry *Type*
    // entity (e.g. "string", "array") asserts the operand's value type. If the
    // resolved op's signature carries none of that type's surface forms, the op
    // was applied to the wrong domain ("reverse a STRING" → `fn reverse(Vec<i64>)`).
    // The type's surface forms come from the registry entity, not a literal table.
    if let Some((type_word, needles)) = mentioned_value_type(input, registry) {
        let satisfied = needles.iter().any(|n| sig_lower.contains(n.as_str()));
        if !satisfied {
            return Some(GateRefusal {
                category: GateCategory::Hard,
                reason: format!(
                    "no operation confidently resolved: request mentions a '{}' value but \
                     resolved op '{}' has signature '{}' (type mismatch)",
                    type_word, req.function_name, req.signature
                ),
            });
        }
    }

    // (3) Operation identity. For each emergently-resolved operation in the
    // request: if it IS the resolved op, the op was genuinely named. If a content
    // word resolves to a DIFFERENT op and the pair is NOT a buildable pipeline
    // (already returned above), the named op was silently dropped — fail closed.
    // If NO content word names the resolved op at all, it was not understood.
    if !ops.is_empty() {
        let mut names_resolved_op = false;
        for op in &ops {
            if op.fn_name == req.function_name {
                names_resolved_op = true;
            } else {
                return Some(GateRefusal {
                    category: GateCategory::CompositionUnsupported,
                    reason: format!(
                        "request also names operation '{}' (resolves to '{}'), dropped in favor \
                         of '{}' — compositional request not yet supported",
                        op.surface, op.fn_name, req.function_name
                    ),
                });
            }
        }
        if !names_resolved_op {
            let surfaces: Vec<&str> = ops.iter().map(|o| o.surface.as_str()).collect();
            return Some(GateRefusal {
                category: GateCategory::Hard,
                reason: format!(
                    "no operation confidently resolved: request content words {:?} do not name the \
                     resolved op '{}'",
                    surfaces, req.function_name
                ),
            });
        }
    } else if !tokenize_lower(input).is_empty() {
        // Tokens present but NONE resolve to any operation (pure operands / gibberish):
        // there is no evidence the resolved op was named. Fail closed.
        return Some(GateRefusal {
            category: GateCategory::Hard,
            reason: format!(
                "no operation confidently resolved: no request token names the resolved op '{}'",
                req.function_name
            ),
        });
    }

    None
}

/// Minimum `ResolutionEvidence.score` for a surface token to count as genuinely
/// naming a programming operation. Coincidental low-confidence links
/// (fuzzy edit-distance ~0.64, definition-overlap ~0.51) fall below this and are
/// treated as operands, exactly as the prior `evidence.method` blocklist intended
/// — but now via the resolver's own numeric confidence, not a method-name switch.
const OP_RESOLVE_FLOOR: f32 = 0.80;

/// A content word that EMERGENTLY resolved to a programming operation.
#[derive(Clone, Debug)]
struct ResolvedContentOp {
    /// The surface token as it appeared in the request.
    surface: String,
    /// The op's canonical function name (`default_fn_name` or lemma).
    fn_name: String,
    /// Declared arity from the registry entity, if any.
    arity: Option<u32>,
    /// `input_types` property (e.g. "i64", "i64,i64", "Vec<i64>"), lowercased.
    input_types: String,
    /// `output_type` property, lowercased.
    output_type: String,
}

/// Resolve every request token to a high-confidence programming operation, in
/// request order, de-duplicated by function name. A token is an *operand* (and
/// silently dropped here) precisely when it does NOT resolve to a
/// Function/Operator entity at or above [`OP_RESOLVE_FLOOR`] — operand-vs-operation
/// is decided by the resolver + entity type, never a wordlist.
/// Minimal score for a token to count as a STATEFUL operand-class signal. Set to
/// the same ARRAY_DOMAIN_FLOOR rationale: a per-tick-state word ("tick", "state",
/// "running", …) links to a stateful op weakly (definition overlap), yet its mere
/// presence — against a plain array-REDUCE primary op — is a real operand-class
/// signal. The behavioral-match step below still gates correctness, so a weak
/// false-positive here cannot produce a confident-wrong solve (it can only fail to
/// behaviorally match and leave the plain reduce untouched).
const STATEFUL_OPERAND_FLOOR: f32 = 0.50;

/// STATEFUL RE-TARGET: detect a per-tick stateful reducer disguised (by the
/// resolver) as a plain array reduce, and re-target the requirement to the mined
/// stateful capability whose engine reducer behaviorally matches. Returns `None`
/// (leaving the plain reduce untouched) unless ALL structural conditions hold:
///
///   1. REDUCE SHAPE (structural, from example types): every resolved example is
///      a single ARRAY input → INT output — i.e. a collection→scalar reduce.
///   2. STATEFUL OPERAND SIGNAL (structural, registry-resolved): some request
///      token resolves (>= floor) to an entity whose `nsynth_category` is
///      `stateful` — the per-tick-state operand class, read from the registry, not
///      a phrase list.
///   3. BEHAVIORAL MATCH (no name table): among the mined stateful entities, pick
///      the one whose engine reducer `g` reproduces the resolved reduce op's
///      output on EVERY row with the additive seed (state=0, op="+" ⇒ f(0,arr) =
///      g(arr) = the reduce op's output). The op identity is chosen by BEHAVIOUR,
///      never by matching the array op's name to a stateful op's name.
fn retarget_stateful_reducer(
    input: &str,
    req: &SynthesisRequirement,
    registry: &Registry,
) -> Option<SynthesisRequirement> {
    use linguigenesis_core::coding_requirements::{parse_example_cases, ExampleSpec};
    use linguigenesis_core::nl_tokens::tokenize_lower;

    // (1) REDUCE SHAPE: collection→scalar. Read from example value types only.
    if req.examples.is_empty() {
        return None;
    }
    let is_reduce_shape = req.examples.iter().all(|ex| {
        ex.inputs.len() == 1
            && matches!(ex.inputs[0], LiteralValue::Array(_))
            && matches!(ex.expected, LiteralValue::Int(_))
    });
    if !is_reduce_shape {
        return None;
    }

    // (2) STATEFUL OPERAND SIGNAL: a request token resolves to a stateful-category
    // entity. The category lives on the registry entity (mined `nsynth_category`),
    // so this is an emergent operand-class read, not a hardcoded phrase list.
    let resolver = EntityResolver::new(registry.clone());
    let has_stateful_operand = tokenize_lower(input).iter().any(|tok| {
        resolver
            .resolve_operation_surface(tok)
            .map(|r| {
                r.evidence.score >= STATEFUL_OPERAND_FLOOR
                    && r.entity.get_property("nsynth_category").map(|c| c.as_str())
                        == Some("stateful")
            })
            .unwrap_or(false)
    });
    if !has_stateful_operand {
        return None;
    }

    // (3) BEHAVIORAL MATCH: pick the mined stateful entity whose additive update
    // (state=0, op="+") reproduces the resolved reduce op's output on every row.
    // f(0, arr) = g(arr) for the matching reducer, so we just need g(arr) == the
    // reduce op's labelled output. We test this via the stateful entity's OWN
    // engine reducer applied at state 0 — sourced from the engine surface through
    // the descriptor, not re-implemented here.
    let reduce_rows: Vec<(&Vec<i64>, i64)> = req
        .examples
        .iter()
        .filter_map(|ex| match (&ex.inputs[0], &ex.expected) {
            (LiteralValue::Array(a), LiteralValue::Int(v)) => Some((a, *v)),
            _ => None,
        })
        .collect();

    let stateful_descriptors = crate::synthesis::stateful_reducer_surface::mineable_stateful_ops();
    for entity in registry.all_entities() {
        if entity.get_property("nsynth_category").map(|c| c.as_str()) != Some("stateful") {
            continue;
        }
        let reducer = entity.get_property("mined_engine_reducer")?.clone();
        let op = entity.get_property("mined_engine_op")?.clone();
        // Confirm this entity is one the descriptor actually emits (bound to the
        // engine surface), and use the descriptor's runner as the reference for
        // `f(state, arr) = state op g(arr)`.
        let Some(desc) = stateful_descriptors.iter().find(|d| {
            d.lemma == entity.lemma && d.reducer == reducer && d.op == op
        }) else {
            continue;
        };
        // EMERGENT left-identity seed: a plain single-array reduce is the special
        // case of the per-tick update where the prior state contributes nothing —
        // i.e. f(e, arr) = e op g(arr) = g(arr) when `e` is op's LEFT-IDENTITY. We
        // derive `e` by PROBING the op's own combine arithmetic
        // (`stateful_state_combine`, the SAME math `search_stateful_reducer`
        // verifies), never a per-op phrase table: `e` is the seed for which
        // `e op x == x` across sample `x`. This generalises the prior op="+"/seed=0
        // case to ALL state-combining ops (running MAX uses e=i64::MIN, running MIN
        // uses e=i64::MAX, a product uses e=1), so the genuinely-stateful
        // non-additive updates become NL-reachable. Ops with NO left-identity (e.g.
        // "-", which never mirrors a plain reduce) yield `None` and are skipped.
        let Some(identity) = op_left_identity(&op) else {
            continue;
        };
        // BEHAVIORAL test: f(e, arr) == the reduce op's labelled output on EVERY
        // row, where `e` is op's left-identity. The op identity is chosen by this
        // BEHAVIOUR (state contributes nothing at the identity seed), not by a name
        // match between the array op and the stateful op.
        let matches_all = reduce_rows
            .iter()
            .all(|(arr, out)| (desc.run)(identity, arr) == Some(*out));
        if !matches_all {
            continue;
        }
        // MATCH. Build a re-targeted requirement from THIS stateful entity's mined
        // (state, arr) example_cases.
        let stateful_examples: Vec<ExampleSpec> = parse_example_cases(&entity);
        if stateful_examples.len() < 3 {
            continue; // keep the holdout-protected floor
        }
        let mut retargeted = req.clone();
        retargeted.function_name = entity.lemma.clone();
        retargeted.category = "stateful".to_string();
        retargeted.target_structure = Some(entity.lemma.clone());
        // Stateful per-tick signature `(state: i64, arr: [i64]) -> i64`, so the
        // downstream gates / `infer_signature` see the real two-input shape rather
        // than the resolver's stale single-input reduce signature.
        retargeted.signature = format!("fn {}(state: i64, arr: [i64]) -> i64", entity.lemma);
        retargeted.examples = stateful_examples;
        retargeted.unresolved.clear();
        retargeted.pipeline = None;
        return Some(retargeted);
    }
    None
}

/// The LEFT-IDENTITY seed `e` of a state-combining `op` (the `op` in
/// `state op g(arr)`): the prior-state value for which the per-tick update
/// degenerates to the plain reduce, `e op x == x` for all `x`. Derived
/// EMERGENTLY by probing the engine's OWN combine arithmetic
/// (`solver::stateful_state_combine`) over a small candidate seed set — NOT a
/// per-op table — so it tracks the engine surface and fails closed (`None`) for
/// an op with no left-identity (e.g. subtraction, which can never mirror a plain
/// reduce). Candidates cover the additive (0), multiplicative (1), and lattice
/// (`i64::MIN` for max, `i64::MAX` for min) identities the 5 stateful ops need.
fn op_left_identity(op: &str) -> Option<i64> {
    // Sample `x` values spanning sign, zero, and magnitude so a false identity
    // (one that happens to fix a single probe) cannot pass.
    const SAMPLES: [i64; 6] = [0, 1, -1, 7, -13, 1_000_000];
    const CANDIDATES: [i64; 4] = [0, 1, i64::MIN, i64::MAX];
    CANDIDATES.into_iter().find(|&e| {
        SAMPLES
            .iter()
            .all(|&x| crate::solver::stateful_state_combine(op, e, x) == Some(x))
    })
}

fn resolved_content_ops(input: &str, registry: &Registry) -> Vec<ResolvedContentOp> {
    use linguigenesis_core::entity::EntityType;
    use linguigenesis_core::nl_tokens::tokenize_lower;

    let resolver = EntityResolver::new(registry.clone());
    let mut ops: Vec<ResolvedContentOp> = Vec::new();
    for tok in tokenize_lower(input) {
        let Some(resolved) = resolver.resolve_operation_surface(&tok) else {
            continue;
        };
        if resolved.evidence.score < OP_RESOLVE_FLOOR {
            continue;
        }
        if !matches!(
            resolved.entity.entity_type,
            EntityType::Function | EntityType::Operator
        ) {
            continue;
        }
        let fn_name = resolved
            .entity
            .get_property("default_fn_name")
            .cloned()
            .unwrap_or_else(|| resolved.entity.lemma.clone());
        let arity = resolved
            .entity
            .get_property("arity")
            .and_then(|s| s.parse::<u32>().ok());
        let input_types = resolved
            .entity
            .get_property("input_types")
            .cloned()
            .unwrap_or_default()
            .to_lowercase();
        let output_type = resolved
            .entity
            .get_property("output_type")
            .cloned()
            .unwrap_or_default()
            .to_lowercase();
        if ops.iter().any(|o| o.fn_name == fn_name) {
            continue;
        }
        ops.push(ResolvedContentOp {
            surface: tok,
            fn_name,
            arity,
            input_types,
            output_type,
        });
    }
    ops
}

/// EMERGENT value-type mention: find the first request token that resolves
/// (high-confidence) to a registry **Type** entity, and return that type's
/// signature surface forms (its lemma + synonyms) so the gate can check the
/// resolved op's signature actually carries the domain. No literal type-noun table.
fn mentioned_value_type(input: &str, registry: &Registry) -> Option<(String, Vec<String>)> {
    use linguigenesis_core::entity::EntityType;
    use linguigenesis_core::nl_tokens::tokenize_lower;

    let resolver = EntityResolver::new(registry.clone());
    for tok in tokenize_lower(input) {
        let Some(resolved) = resolver.resolve_surface(&tok) else {
            continue;
        };
        if resolved.entity.entity_type != EntityType::Type {
            continue;
        }
        if resolved.evidence.score < OP_RESOLVE_FLOOR {
            continue;
        }
        // The type's accepted signature surface forms: its lemma plus any declared
        // `signature_aliases` property (comma-separated) — emergent from the entity.
        let mut needles: Vec<String> = vec![resolved.entity.lemma.to_lowercase()];
        if let Some(aliases) = resolved.entity.get_property("signature_aliases") {
            for a in aliases.split(',') {
                let a = a.trim().to_lowercase();
                if !a.is_empty() {
                    needles.push(a);
                }
            }
        }
        return Some((tok, needles));
    }
    None
}

/// Minimal resolution score for a token to count as *evidence of the array
/// domain*. This is intentionally BELOW [`OP_RESOLVE_FLOOR`]: an array-context
/// word like "array" links to an array op only weakly (definition-overlap), yet
/// its mere presence — against a SCALAR resolved op — is a real domain signal.
/// Pure non-resolving noise (score 0 / no match) never reaches here.
const ARRAY_DOMAIN_FLOOR: f32 = 0.50;
/// Minimum CONTENT-resolution confidence for a token to count as a real array
/// operand. A registered domain noun ("array") resolves at ~1.0; a meta word
/// ("function") only fuzzy-links via WordNet below this, so it is not misread as
/// an operand even when the grammar-marker classification misses it.
const ARRAY_CONTENT_FLOOR: f32 = 0.9;

/// Find the first request token that resolves to an ARRAY-domain operation (an op
/// whose declared `input_types` contains a vector type) other than `req_fn`.
/// Returns the surface word so the gate can report the domain mismatch. Emergent:
/// the operand domain is read from the resolved entity's signature, not a list.
/// Solve `problem`, then re-verify the emitted code against the FULL spec
/// (examples + holdouts). `problem_from_requirement` reserves the last distinct
/// row as a fresh holdout, and `solve_problem` solves a `synthesis_view()` that
/// CLEARS holdouts — so the holdout never bites INSIDE the solver. Without this
/// post-solve re-check an overfit that fits the seed but contradicts the holdout
/// is wrongly accepted (e.g. "trim" synthesized as remove-ALL-spaces, which fits
/// the leading/trailing-space seed rows but fails "a b c" -> "a b c"). This is the
/// SOUNDNESS gate for the single-op NL door — mirrors the LLM-examples path.
fn solve_verifying_holdouts(problem: &crate::benchmark::Problem) -> crate::solver::SolveResult {
    let res = crate::solver::solve_problem(problem);
    if res.success && !problem.holdouts.is_empty() {
        let full: Vec<crate::benchmark::Example> = problem
            .examples
            .iter()
            .chain(problem.holdouts.iter())
            .cloned()
            .collect();
        if !crate::runtime::code_reproduces_examples(&res.code, &full) {
            // The seed-fit is an OVERFIT — it contradicts the holdout (e.g. "trim"
            // as remove-ALL-spaces, which fits the leading/trailing seed rows but
            // fails "a b c" -> "a b c"). COMPLETENESS: re-solve with the holdout
            // FOLDED INTO the examples so the solver's own verification rejects the
            // overfit and continues to a GENERALIZING program (string_synth's real
            // s.trim()). The retry is re-checked against the full spec, so
            // acceptance stays sound — only a program that reproduces EVERY row wins.
            let mut full_problem = problem.clone();
            full_problem.examples = full.clone();
            full_problem.holdouts = Vec::new();
            let res2 = crate::solver::solve_problem(&full_problem);
            if res2.success && crate::runtime::code_reproduces_examples(&res2.code, &full) {
                return res2;
            }
            return crate::solver::SolveResult {
                success: false,
                code: res.code,
                method: res.method,
                error: Some("solved code fails the held-out probe (overfit)".to_string()),
                metadata: Default::default(),
            };
        }
    }
    res
}

fn array_domain_word(
    input: &str,
    registry: &Registry,
    req_fn: &str,
    req_sig: &str,
) -> Option<String> {
    use linguigenesis_core::entity::EntityType;
    use linguigenesis_core::nl_tokens::tokenize_lower;

    let resolver = EntityResolver::new(registry.clone());
    // A token that merely names the resolved op's OWN type — "string" for a
    // `string -> string` op, "array" for a genuine array op — is CONSISTENT with
    // the op, not a foreign-domain operand, so it must not trip the array/scalar
    // mismatch guard. The resolved op's signature carries its type words, so skip
    // any operand token that appears in it (emergent, no hardcoded type list).
    // This fixes the false positive where "trim a string" was rejected because
    // "string" fuzzily op-links to a vec-input op.
    let sig_l = req_sig.to_lowercase();
    for tok in tokenize_lower(input) {
        if tok.len() >= 3 && sig_l.contains(tok.as_str()) {
            continue; // token names a type in the op's own signature — not a mismatch
        }
        // The token must be a genuine DOMAIN word, not a structural / stop word.
        // Consult the DATA stop set directly (coding_registry grammar_stop_words):
        // meta words like "function"/"number" must never count as domain evidence.
        // (Previously this leaned on resolve_surface returning a GrammarMarker for
        // stop words, but a content entity at the same 0.51 tier can outcompete it
        // nondeterministically — e.g. "function"→map via definition overlap once
        // the merge stopped losing entities.)
        if resolver.is_stop_word(&tok) {
            continue;
        }
        // `resolve_operation_surface` fuzzily links almost ANY token to SOME op at
        // ~0.51 (e.g. "function"→map, "array"→array_max BOTH score 0.51), so the
        // weak op-link alone cannot tell a real array operand ("array") from meta
        // noise ("a FUNCTION that…"). We disambiguate emergently via the token's
        // CONTENT resolution (`resolve_surface`): a real domain operand resolves to
        // a high-confidence Noun/Type entity (registry "array" → Noun, score 1.0),
        // whereas a stop word resolves to a GrammarMarker (registry "function" →
        // GrammarMarker "grammar_stop_words", 0.51) or to nothing content-ful.
        // Skip the latter. This reads the registry's own entity types/scores — no
        // hardcoded stop-word list here.
        match resolver.resolve_surface(&tok) {
            Some(content) => {
                if matches!(
                    content.entity.entity_type,
                    EntityType::GrammarMarker | EntityType::ConstraintMarker
                ) {
                    continue;
                }
                // A genuine array OPERAND is a REGISTERED domain word ("array" ->
                // noun, content score ~1.0). A META word ("function", "method")
                // only fuzzy-links via WordNet at a low content score. Require a
                // high-confidence CONTENT resolution so a meta word — which the
                // grammar-marker classification no longer catches after registry
                // churn — is not misread as an array operand. Emergent: reads the
                // resolver's own confidence, no stop-word list.
                if content.evidence.score < ARRAY_CONTENT_FLOOR {
                    continue;
                }
            }
            None => continue,
        }
        let Some(resolved) = resolver.resolve_operation_surface(&tok) else {
            continue;
        };
        if resolved.evidence.score < ARRAY_DOMAIN_FLOOR {
            continue;
        }
        if !matches!(
            resolved.entity.entity_type,
            EntityType::Function | EntityType::Operator
        ) {
            continue;
        }
        let in_types = resolved
            .entity
            .get_property("input_types")
            .cloned()
            .unwrap_or_default()
            .to_lowercase();
        if !in_types.contains("vec") {
            continue;
        }
        let fname = resolved
            .entity
            .get_property("default_fn_name")
            .cloned()
            .unwrap_or_else(|| resolved.entity.lemma.clone());
        if fname != req_fn {
            return Some(tok);
        }
    }
    None
}

/// The associative fold a reduce primitive computes, determined by EXECUTING the
/// synthesized reduce code on probe arrays — never inferred from the op's name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoldKind {
    /// `acc = acc + x`, identity 0.
    Sum,
    /// `acc = acc * x`, identity 1.
    Product,
    /// running maximum (seeded with the first element).
    Max,
    /// running minimum (seeded with the first element).
    Min,
}

/// The array→array reordering an ArrayTransform primitive computes, determined by
/// EXECUTING the synthesized transform code on probe arrays — never inferred from
/// the op's name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArrayTransformKind {
    /// Sort ascending.
    Sort,
    /// Reverse element order.
    Reverse,
}

/// Classify an ARRAY-transform primitive (`Vec<i64> -> Vec<i64>`) by running its
/// SYNTHESIZED code on probe arrays and matching the output against each candidate
/// transform's known result. Behaviour-driven: the op's NAME is only a label, so
/// "is it sort vs reverse" is decided by execution, not assumption. Returns `None`
/// when neither candidate matches (e.g. the synthesized code is an affine misfit
/// of a single-example op) so the caller can fall back to the registry spec.
fn classify_array_transform_by_exec(transform_code: &str, transform_fn: &str) -> Option<ArrayTransformKind> {
    use crate::benchmark::{Problem, Value};
    let problem = Problem {
        name: "arrxfm_probe".to_string(),
        category: "probe",
        description: "array transform classification",
        signature: "fn f(a: [i64]) -> [i64]",
        examples: vec![],
        ..Default::default()
    };
    // Probes where sort and reverse disagree (already-sorted/reversed inputs would
    // not discriminate), and where identity differs from both.
    let probes: &[&[i64]] = &[&[3, 1, 2], &[5, 2, 8, 1], &[4, 7, 3, 9, 1]];
    let run = |arr: &[i64]| -> Option<Vec<i64>> {
        match crate::runtime::execute_function_for_problem(
            transform_code,
            transform_fn,
            &[Value::int_array(arr)],
            &problem,
        ) {
            Ok(crate::runtime::Value::Array(vs)) => vs
                .iter()
                .map(|v| match v {
                    crate::runtime::Value::Int(n) => Some(*n),
                    _ => None,
                })
                .collect(),
            _ => None,
        }
    };
    let candidates = [ArrayTransformKind::Sort, ArrayTransformKind::Reverse];
    for cand in candidates {
        let mut all_match = true;
        for p in probes {
            let mut expected: Vec<i64> = p.to_vec();
            match cand {
                ArrayTransformKind::Sort => expected.sort(),
                ArrayTransformKind::Reverse => expected.reverse(),
            }
            if run(p).as_deref() != Some(expected.as_slice()) {
                all_match = false;
                break;
            }
        }
        if all_match {
            return Some(cand);
        }
    }
    None
}

/// Accepted, strict-verified transform-chain pipeline result.
#[derive(Clone, Debug)]
pub struct PipelineOutcome {
    /// The original NL request.
    pub description: String,
    /// The synthesized pipeline's own function name (call this to run `code`).
    pub fn_name: String,
    /// Element map op function names in REQUEST ORDER (outer→inner). Empty for a
    /// reduce-only pipeline.
    pub map_fns: Vec<String>,
    /// Array-transform op function name, if the pipeline has a sort/reverse stage
    /// between the map chain and any reduce. `None` when no array transform.
    pub array_xfm_fns: Vec<String>,
    /// The array transform kinds (behaviour-classified), in chain order.
    pub array_xfms: Vec<ArrayTransformKind>,
    /// Aggregate (reduce) op function name, if the pipeline has a reduce stage
    /// (shape a). `None` for an array-output map chain (shape b).
    pub reduce_fn: Option<String>,
    /// The fold kind the reduce computes (behaviour-classified). `None` for an
    /// array-output map chain (no reduce).
    pub fold: Option<FoldKind>,
    /// The solver-synthesized program for the whole pipeline.
    pub code: String,
    /// `nl-compose-chain:<inner-solver-method>`.
    pub method: String,
}

impl PipelineOutcome {
    /// True iff this is a genuine multi-stage pipeline (>=1 map and a reduce, or
    /// a chain of >=2 maps) rather than a coincidental single-op solve. Used by
    /// the accept-test to reject a single-op match masquerading as a composition.
    pub fn is_two_stage(&self) -> bool {
        // Any reduce-bearing pipeline (shape a — including reduce-on-implicit-
        // array), a chain of >=2 maps, or a map+array-transform / standalone
        // non-req array transform is genuinely multi-stage. A lone single map
        // with no reduce never reaches acceptance (classify_pipeline returns None
        // for the plain single op).
        self.reduce_fn.is_some() || self.map_fns.len() >= 2 || !self.array_xfm_fns.is_empty()
    }

    /// True iff this pipeline contains a genuine array-transform stage (sort /
    /// reverse) composed over a map chain — the >=2-stage array→array shape the
    /// NL-COMPOSE-ARRTRANSFORM accept-criterion requires.
    pub fn has_array_transform(&self) -> bool {
        !self.array_xfm_fns.is_empty()
    }

    /// Length of the element-transform chain (number of composed ScalarMaps).
    pub fn map_chain_len(&self) -> usize {
        self.map_fns.len()
    }

    pub(crate) fn into_solve_result(self) -> crate::solver::SolveResult {
        crate::solver::SolveResult {
            success: true,
            code: self.code,
            method: self.method,
            error: None,
            metadata: Default::default(),
        }
    }
}

/// Deterministic function name for a pipeline, so the strict verifier's holdout
/// seed (which hashes `problem.name`) is stable.
fn pipeline_fn_name(plan: &CompositionPlan) -> String {
    let mut name = String::from("compose");
    if let Some(r) = &plan.reduce {
        name.push('_');
        name.push_str(&r.fn_name);
    } else {
        name.push_str("_maps");
    }
    for t in &plan.array_transforms {
        name.push('_');
        name.push_str(&t.fn_name);
    }
    if let Some(f) = &plan.filter {
        name.push_str("_filter_");
        name.push_str(&f.word);
    }
    for m in &plan.maps {
        name.push('_');
        name.push_str(&m.fn_name);
    }
    name
}

fn describe_plan(plan: &CompositionPlan) -> String {
    let maps: Vec<&str> = plan.maps.iter().map(|m| m.fn_name.as_str()).collect();
    let chain = if maps.is_empty() {
        "(no map)".to_string()
    } else {
        maps.join(" ∘ ")
    };
    let xfm = if plan.array_transforms.is_empty() {
        String::new()
    } else {
        let names: Vec<&str> = plan.array_transforms.iter().map(|t| t.fn_name.as_str()).collect();
        format!(" arrayxfm={}", names.join("→"))
    };
    let filt = match &plan.filter {
        Some(f) => format!(" filter={}({} {})", f.word, f.cmp, f.value),
        None => String::new(),
    };
    match &plan.reduce {
        Some(r) => format!("reduce={} ∘{}{} mapchain=[{}]", r.fn_name, filt, xfm, chain),
        None => format!("mapchain=[{}]{}{} -> array (no reduce)", chain, filt, xfm),
    }
}

/// Classify an ARRAY-reduce primitive's fold by running it on probe arrays and
/// matching the output against each candidate fold's known result. Behaviour-
/// driven: the op's NAME is only a label in error messages.
fn classify_array_fold(reduce_code: &str, reduce_fn: &str) -> Option<FoldKind> {
    use crate::benchmark::{Problem, Value};
    let problem = Problem {
        name: "reduce_probe".to_string(),
        category: "probe",
        description: "fold classification",
        signature: "fn f(a: [i64]) -> i64",
        examples: vec![],
        ..Default::default()
    };
    // Distinct probes so the four folds give DIFFERENT answers (no collision).
    let probes: &[&[i64]] = &[&[3, 1, 2], &[2, 5, 4], &[2, 3, 4]];
    let run = |arr: &[i64]| -> Option<i64> {
        match crate::runtime::execute_function_for_problem(
            reduce_code,
            reduce_fn,
            &[Value::int_array(arr)],
            &problem,
        ) {
            Ok(crate::runtime::Value::Int(v)) => Some(v),
            _ => None,
        }
    };
    let candidates = [FoldKind::Sum, FoldKind::Product, FoldKind::Max, FoldKind::Min];
    for cand in candidates {
        let mut all_match = true;
        for p in probes {
            let expected = match cand {
                FoldKind::Sum => p.iter().sum::<i64>(),
                FoldKind::Product => p.iter().product::<i64>(),
                FoldKind::Max => *p.iter().max().unwrap(),
                FoldKind::Min => *p.iter().min().unwrap(),
            };
            if run(p) != Some(expected) {
                all_match = false;
                break;
            }
        }
        if all_match {
            return Some(cand);
        }
    }
    None
}

/// Classify a BINARY scalar reduce-seed (e.g. `add`/`multiply`) by running it on
/// scalar PROBE PAIRS to learn its combiner — `a+b` ⇒ Sum-fold, `a*b` ⇒
/// Product-fold. Behaviour-driven (the op is executed), never name-keyed. A
/// binary op whose behaviour is neither addition nor multiplication is not a
/// supported fold seed (returns `None`).
fn classify_binary_fold(reduce_code: &str, reduce_fn: &str) -> Option<FoldKind> {
    use crate::benchmark::{Problem, Value};
    let problem = Problem {
        name: "binop_probe".to_string(),
        category: "probe",
        description: "binary fold classification",
        signature: "fn f(a: i64, b: i64) -> i64",
        examples: vec![],
        ..Default::default()
    };
    let run = |a: i64, b: i64| -> Option<i64> {
        match crate::runtime::execute_function_for_problem(
            reduce_code,
            reduce_fn,
            &[Value::Int(a), Value::Int(b)],
            &problem,
        ) {
            Ok(crate::runtime::Value::Int(v)) => Some(v),
            _ => None,
        }
    };
    // Probe pairs chosen so + and * disagree on every pair.
    let pairs = [(2, 3), (4, 5), (1, 7), (3, 6)];
    let is_sum = pairs.iter().all(|&(a, b)| run(a, b) == Some(a + b));
    if is_sum {
        return Some(FoldKind::Sum);
    }
    let is_prod = pairs.iter().all(|&(a, b)| run(a, b) == Some(a * b));
    if is_prod {
        return Some(FoldKind::Product);
    }
    None
}

/// Emit an INDEPENDENT reference implementation of the pipeline in the runtime
/// DSL. The map chain's fn bodies (verified single-op synthesis) go first so the
/// driver can call them; the element expression nests the chain in REQUEST ORDER
/// (`maps[0](maps[1](...maps[n-1](arr[i])))`, outer→inner). Then:
///   * if an `array_xfm` is present, the driver MATERIALIZES the mapped array,
///     applies the sort/reverse to it, and finally folds it (shape a) or returns
///     it (shape b) — the array transform sits between the map chain and reduce;
///   * otherwise the driver folds the element expression inline (shape a) or
///     builds the mapped array (shape b), exactly as before.
/// Used only to LABEL fresh holdouts; the accepted program is what the solver finds.
/// Emit the composed REFERENCE for a FILTER pipeline: map-chain applied to each
/// element, kept only when the predicate holds, then either summed/multiplied
/// (Sum/Product reduce) or returned as the filtered array. Independent impl used
/// only to label fresh holdouts — the solver synthesizes the real program.
fn emit_filter_pipeline_reference(
    composed_name: &str,
    fold: Option<FoldKind>,
    filter: &FilterPred,
    map_chain: &[(String, String)],
) -> String {
    let mut out = String::new();
    let mut emitted: Vec<&str> = Vec::new();
    for (map_fn, map_code) in map_chain {
        if emitted.contains(&map_fn.as_str()) {
            continue;
        }
        emitted.push(map_fn.as_str());
        out.push_str(map_code);
        if !out.ends_with('\n') {
            out.push('\n');
        }
        out.push('\n');
    }
    // elem(item): nest the map chain outer→inner around the raw element.
    let mut elem_item = "item".to_string();
    for (map_fn, _) in map_chain.iter().rev() {
        elem_item = format!("{}({})", map_fn, elem_item);
    }
    let cmp = match filter.cmp.as_str() {
        "gt" => ">",
        "lt" => "<",
        "ge" => ">=",
        "le" => "<=",
        "eq" => "==",
        _ => "!=",
    };
    let cond = match filter.modulus {
        Some(m) => format!("e % {} {} {}", m, cmp, filter.value),
        None => format!("e {} {}", cmp, filter.value),
    };
    match fold {
        Some(fk) => {
            let (init, op) = match fk {
                FoldKind::Product => (1, "*"),
                _ => (0, "+"), // Sum (Max/Min+filter excluded upstream)
            };
            out.push_str(&format!(
                "fn {composed_name}(arr: [i64]) -> i64 {{\n    acc: i64 = {init};\n    \
                 for item in arr {{\n        e: i64 = {elem_item};\n        if {cond} {{\n            \
                 acc = acc {op} e;\n        }}\n    }}\n    return acc;\n}}\n"
            ));
        }
        None => {
            out.push_str(&format!(
                "fn {composed_name}(arr: [i64]) -> [i64] {{\n    result: [i64] = [];\n    \
                 for item in arr {{\n        e: i64 = {elem_item};\n        if {cond} {{\n            \
                 result.push(e);\n        }}\n    }}\n    return result;\n}}\n"
            ));
        }
    }
    out
}

/// Emit Mog applying an ordered array-transform CHAIN to `start_var`; returns
/// (code, final_var_name). Sort mutates in place (var unchanged); reverse builds a
/// fresh array. Stage-indexed names (`xfm{i}`/`k{i}`) avoid collisions across a
/// multi-transform chain. A single transform emits code equivalent to the prior
/// single-transform path.
fn emit_transform_chain(start_var: &str, xfms: &[ArrayTransformKind]) -> (String, String) {
    let mut code = String::new();
    let mut cur = start_var.to_string();
    for (i, kind) in xfms.iter().enumerate() {
        match kind {
            ArrayTransformKind::Sort => {
                code.push_str(&format!("    {cur}.sort();\n"));
            }
            ArrayTransformKind::Reverse => {
                let next = format!("xfm{i}");
                code.push_str(&format!(
                    "    {next}: [i64] = [];\n    k{i}: i64 = {cur}.len - 1;\n    \
                     while k{i} >= 0 {{\n        {next}.push({cur}[k{i}]);\n        \
                     k{i} = k{i} - 1;\n    }}\n"
                ));
                cur = next;
            }
        }
    }
    (code, cur)
}

fn emit_pipeline_reference(
    composed_name: &str,
    fold: Option<FoldKind>,
    array_xfms: &[ArrayTransformKind],
    map_chain: &[(String, String)],
) -> String {
    let mut out = String::new();
    // Emit each distinct map fn body once (the chain may legitimately repeat an
    // op, but the body only needs to appear a single time).
    let mut emitted: Vec<&str> = Vec::new();
    for (map_fn, map_code) in map_chain {
        if emitted.contains(&map_fn.as_str()) {
            continue;
        }
        emitted.push(map_fn.as_str());
        out.push_str(map_code);
        if !out.ends_with('\n') {
            out.push('\n');
        }
        out.push('\n');
    }

    // Nest the chain outer→inner around the element: maps[0]( ... maps[n-1](inner) ).
    let elem = |inner: &str| -> String {
        let mut expr = inner.to_string();
        for (map_fn, _) in map_chain.iter().rev() {
            expr = format!("{}({})", map_fn, expr);
        }
        expr
    };

    // ── ARRAY-TRANSFORM present: materialize mapped array, reorder, then fold or
    //    return. The reorder uses the SAME DSL the dedicated sort/reverse
    //    array_transform candidates emit (`mapped.sort()` / index-loop reverse). ──
    // ── ARRAY-TRANSFORM chain present: materialize the mapped array, apply each
    //    transform in request order (sort then reverse, …), then fold to a scalar
    //    or return the array. Uses the SAME DSL the dedicated sort/reverse
    //    candidates emit, via `emit_transform_chain`. ──
    if !array_xfms.is_empty() {
        let elem_item = elem("item");
        let build = format!(
            "    mapped: [i64] = [];\n    for item in arr {{\n        mapped.push({elem_item});\n    }}\n"
        );
        let (chain, final_var) = emit_transform_chain("mapped", array_xfms);
        match fold {
            // Shape (a): reduce over the transformed array.
            Some(fk) => {
                let body = emit_fold_over_named_array(fk, &final_var);
                out.push_str(&format!(
                    "fn {composed_name}(arr: [i64]) -> i64 {{\n{build}{chain}{body}}}\n"
                ));
            }
            // Shape (b): return the transformed array.
            None => {
                out.push_str(&format!(
                    "fn {composed_name}(arr: [i64]) -> [i64] {{\n{build}{chain}    return {final_var};\n}}\n"
                ));
            }
        }
        return out;
    }

    match fold {
        // ── Shape (a): reduce present → fold the mapped element to a scalar. ──
        Some(FoldKind::Sum) | Some(FoldKind::Product) => {
            let elem = elem("arr[i]");
            let (init, op) = match fold {
                Some(FoldKind::Sum) => (0, "+"),
                Some(FoldKind::Product) => (1, "*"),
                _ => unreachable!(),
            };
            out.push_str(&format!(
                "fn {name}(arr: [i64]) -> i64 {{\n    \
                 acc: i64 = {init};\n    \
                 i: i64 = 0;\n    \
                 while i < arr.len {{\n        \
                 acc = acc {op} {elem};\n        \
                 i = i + 1;\n    }}\n    \
                 return acc;\n}}\n",
                name = composed_name,
                init = init,
                op = op,
                elem = elem,
            ));
        }
        Some(FoldKind::Max) | Some(FoldKind::Min) => {
            let elem = elem("arr[i]");
            let cmp = if fold == Some(FoldKind::Max) { ">" } else { "<" };
            out.push_str(&format!(
                "fn {name}(arr: [i64]) -> i64 {{\n    \
                 i: i64 = 0;\n    \
                 acc: i64 = {elem};\n    \
                 i = 1;\n    \
                 while i < arr.len {{\n        \
                 v: i64 = {elem};\n        \
                 if v {cmp} acc {{ acc = v; }}\n        \
                 i = i + 1;\n    }}\n    \
                 return acc;\n}}\n",
                name = composed_name,
                elem = elem,
                cmp = cmp,
            ));
        }
        // ── Shape (b): no reduce → build the mapped array (array output). ─────
        None => {
            let elem = elem("item");
            out.push_str(&format!(
                "fn {name}(arr: [i64]) -> [i64] {{\n    \
                 result: [i64] = [];\n    \
                 for item in arr {{\n        \
                 result.push({elem});\n    }}\n    \
                 return result;\n}}\n",
                name = composed_name,
                elem = elem,
            ));
        }
    }
    out
}

/// Emit a fold body (Sum/Product/Max/Min) over an already-built array variable
/// `arr_name`, returning the scalar. Used by the array-transform reference path
/// where the array has already been mapped + reordered into `arr_name`. The
/// emitted block ENDS with `return acc;` and is wrapped by the caller's `fn`.
fn emit_fold_over_named_array(fold: FoldKind, arr_name: &str) -> String {
    match fold {
        FoldKind::Sum | FoldKind::Product => {
            let (init, op) = match fold {
                FoldKind::Sum => (0, "+"),
                FoldKind::Product => (1, "*"),
                _ => unreachable!(),
            };
            format!(
                "    acc: i64 = {init};\n    j: i64 = 0;\n    while j < {arr}.len {{\n        acc = acc {op} {arr}[j];\n        j = j + 1;\n    }}\n    return acc;\n",
                arr = arr_name
            )
        }
        FoldKind::Max | FoldKind::Min => {
            let cmp = if fold == FoldKind::Max { ">" } else { "<" };
            format!(
                "    acc: i64 = {arr}[0];\n    j: i64 = 1;\n    while j < {arr}.len {{\n        v: i64 = {arr}[j];\n        if v {cmp} acc {{ acc = v; }}\n        j = j + 1;\n    }}\n    return acc;\n",
                arr = arr_name
            )
        }
    }
}

fn synthesis_requirement_to_examples(
    req: &SynthesisRequirement,
) -> Result<Vec<Example>, BridgeError> {
    req.examples
        .iter()
        .map(|spec| {
            Ok(Example {
                inputs: spec
                    .inputs
                    .iter()
                    .map(literal_to_value)
                    .collect::<Result<Vec<_>, _>>()?,
                expected: literal_to_value(&spec.expected)?,
            })
        })
        .collect()
}

fn literal_to_value(lit: &LiteralValue) -> Result<Value, BridgeError> {
    Ok(match lit {
        LiteralValue::Int(v) => Value::Int(*v),
        LiteralValue::Float(v) => Value::Float(v.to_bits()),
        LiteralValue::Str(s) => Value::Str(s.clone()),
        LiteralValue::Bool(b) => Value::Bool(*b),
        LiteralValue::Array(a) => Value::int_array(a),
        LiteralValue::Pair(a, b) => Value::Pair(*a, *b),
        // Named-field struct: fields convert recursively (flat structs of
        // scalar/array fields; the parser canonicalized field order).
        LiteralValue::Struct(fields) => Value::Struct(
            fields
                .iter()
                .map(|(n, v)| Ok((n.clone(), literal_to_value(v)?)))
                .collect::<Result<Vec<_>, BridgeError>>()?,
        ),
    })
}

/// Infer function signature from examples
pub fn infer_signature(fn_name: &str, examples: &[Example]) -> String {
    if examples.is_empty() {
        return format!("fn {}() -> i64", fn_name);
    }

    let first = &examples[0];
    let mut param_types = Vec::new();
    let mut param_idx = 0;

    for input in &first.inputs {
        let type_str = match input {
            Value::Int(_) => "i64",
            Value::Float(_) => "f64",
            Value::Str(_) => "String",
            Value::Bool(_) => "bool",
            Value::Array(_) => "[i64]",
            Value::Pair(_, _) => "(i64, i64)",
            Value::Quad(_, _, _, _) => "{a: i64, b: i64, c: i64, d: i64}",
            Value::Tree(_) => "Tree",
            Value::Tuple(_) => "Tuple",
            Value::Struct(_) => "Struct",
            Value::Tensor { .. } => "Tensor",
            // A map INPUT is handed to Mog code as an array of [key, value]
            // pairs (the same shape the output bridge verifies), so the declared
            // param type is the nested-array form.
            Value::Map(_) => "[[i64]]",
        };
        param_idx += 1;
        param_types.push(format!("{}: {}", param_names(param_idx), type_str));
    }

    let return_type = match &first.expected {
        Value::Int(_) => "i64",
        Value::Float(_) => "f64",
        Value::Str(_) => "String",
        Value::Bool(_) => "bool",
        Value::Array(_) => "[i64]",
        Value::Pair(_, _) => "(i64, i64)",
        Value::Quad(_, _, _, _) => "{a: i64, b: i64, c: i64, d: i64}",
        Value::Tree(_) => "Tree",
        Value::Tuple(_) => "Tuple",
        Value::Struct(_) => "Struct",
        Value::Tensor { .. } => "Tensor",
        // Map RETURN: the program emits an array of [key, value] pairs; the
        // verifier's array-of-pairs bridge compares it to the expected Map
        // order-independently.
        Value::Map(_) => "Map",
    };

    format!(
        "fn {}({}) -> {}",
        fn_name,
        param_types.join(", "),
        return_type
    )
}

fn param_names(idx: usize) -> String {
    let names = ["a", "b", "c", "d", "e", "f", "g", "h"];
    if idx <= names.len() {
        names[idx - 1].to_string()
    } else {
        format!("arg{}", idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The contract lane is inert (returns None) when the project gate is unset, so
    /// the untrusted LLM door never opens implicitly.
    #[test]
    fn synthesize_project_with_contracts_inert_without_flag() {
        std::env::remove_var("NSYNTH_LOCAL_LLM_PROJECT");
        assert!(LinguigenesisBridge::default()
            .synthesize_project_with_contracts("build a list utility")
            .is_none());
    }

    /// The JSON→Value coercion accepts exactly the supported kinds and refuses
    /// anything it cannot represent soundly (floats, mixed arrays) rather than
    /// coercing them into a wrong runtime value.
    #[test]
    fn json_to_bench_value_maps_supported_kinds() {
        use crate::benchmark::Value;
        assert_eq!(json_to_bench_value(&serde_json::json!(true)), Some(Value::Bool(true)));
        assert_eq!(json_to_bench_value(&serde_json::json!(7)), Some(Value::Int(7)));
        assert_eq!(
            json_to_bench_value(&serde_json::json!([1, 2, 3])),
            Some(Value::int_array(&[1, 2, 3]))
        );
        // Mixed array: the all-int collect short-circuits to None, and a JSON array
        // is not a string, so the whole maps to None (not a coerced Str).
        assert!(json_to_bench_value(&serde_json::json!([1, "x"])).is_none());
        // Float is not bool/i64/array/str → None.
        assert!(json_to_bench_value(&serde_json::json!(1.5)).is_none());
    }

    /// UNWALL-1B EMERGENT left-identity derivation: the non-additive stateful
    /// re-target picks each op's behavioral seed by deriving its LEFT-IDENTITY from
    /// the engine's OWN combine arithmetic (`solver::stateful_state_combine`), never
    /// a per-op phrase table. This asserts the broadened match relies on the real
    /// identities — `+`→0, `*`→1, `max`→i64::MIN, `min`→i64::MAX — and that an op
    /// with NO left-identity (subtraction) yields `None` so it can never mirror a
    /// plain reduce (the additive-only behaviour, generalised soundly).
    #[test]
    fn op_left_identity_is_engine_derived_per_op() {
        assert_eq!(op_left_identity("+"), Some(0), "+ identity is 0");
        assert_eq!(op_left_identity("*"), Some(1), "* identity is 1");
        assert_eq!(op_left_identity("max"), Some(i64::MIN), "max identity is i64::MIN");
        assert_eq!(op_left_identity("min"), Some(i64::MAX), "min identity is i64::MAX");
        // Subtraction has no left-identity (0 - x = -x != x), so a non-additive
        // request can never wrongly mirror a plain reduce via `-`.
        assert_eq!(op_left_identity("-"), None, "subtraction has no left-identity");
        // An op absent from the engine surface fails closed.
        assert_eq!(op_left_identity("xor"), None, "unknown op fails closed");
        // The derived identity actually degenerates the engine update to the plain
        // reduce: f(e, arr) = e op g(arr) = g(arr). Probe max + min directly.
        let arr = &[3i64, -1, 7, 2];
        let id_max = op_left_identity("max").unwrap();
        assert_eq!(
            crate::solver::stateful_reducer_apply("max", "max", id_max, arr),
            Some(7),
            "max left-identity seed must degenerate state.max(max(arr)) to max(arr)"
        );
        let id_min = op_left_identity("min").unwrap();
        assert_eq!(
            crate::solver::stateful_reducer_apply("min", "min", id_min, arr),
            Some(-1),
            "min left-identity seed must degenerate state.min(min(arr)) to min(arr)"
        );
    }

    /// NL-COMPREHEND-IN-LGCORE ANTI-CHEAT: linguigenesis-core — NOT this bridge —
    /// now COMPREHENDS the composition. Calling `CodingComprehension::comprehend`
    /// directly (the lg-core entry point) on a compositional request must return a
    /// `SynthesisRequirement` whose `pipeline` field is `Some(reduce=sum, maps=
    /// [negate])`. The bridge only EXECUTES this emitted plan; it derives nothing.
    /// (This mirrors the lg-core-side unit test `comprehend_emits_pipeline_for_sum_
    /// of_negated`, but runs under the nsynth build which links lg-core as a path
    /// dependency, so the relocated comprehension is proven end-to-end here too.)
    #[test]
    fn lgcore_comprehend_emits_composition_plan() {
        let registry = LinguigenesisBridge::new()
            .registry_clone()
            .expect("registry");
        let mut coding = CodingComprehension::new(registry);
        let req = coding.comprehend("sum of the negated values");
        let plan = req
            .pipeline
            .as_ref()
            .unwrap_or_else(|| panic!("lg-core must emit pipeline; unresolved={:?}", req.unresolved));
        assert_eq!(plan.maps.len(), 1, "maps={:?}", plan.maps);
        assert_eq!(plan.maps[0].role, OpRole::ScalarMap);
        assert_eq!(plan.maps[0].fn_name, "negate");
        let reduce = plan.reduce.as_ref().expect("reduce stage");
        assert!(
            matches!(reduce.role, OpRole::ArrayReduce | OpRole::BinaryFoldSeed),
            "reduce role={:?}",
            reduce.role
        );
        assert!(
            reduce.fn_name.contains("sum") || reduce.fn_name == "add",
            "reduce fn_name={}",
            reduce.fn_name
        );
        // Single-op requests must comprehend with NO pipeline (unchanged path).
        for single in ["square a number", "compute the total of an array"] {
            let r = coding.comprehend(single);
            assert!(
                r.pipeline.is_none(),
                "{single:?} must have no pipeline, got {:?}",
                r.pipeline
            );
        }
    }

    /// NL-COMPREHEND-IN-LGCORE STRUCTURAL PROOF: the comprehended plan is what
    /// drives acceptance — a composed program built from the emitted plan must
    /// strict-verify on FRESH holdouts. (Distinct from the integration suite: this
    /// asserts the bridge consumed `req.pipeline`, not a bridge-local derivation.)
    #[test]
    fn lgcore_plan_drives_strict_verified_composition() {
        let bridge = LinguigenesisBridge::new();
        let outcome = bridge
            .try_compose_pipeline("sum of the negated values")
            .expect("recognised as pipeline (plan came from lg-core comprehend)")
            .expect("built and strict-verified");
        assert!(outcome.is_two_stage(), "must be a genuine multi-stage pipeline");
        assert_eq!(outcome.map_fns, vec!["negate".to_string()]);
        assert!(outcome.reduce_fn.is_some(), "reduce stage present");
        assert!(
            outcome.method.starts_with("nl-compose-chain:"),
            "method tag={}",
            outcome.method
        );
    }

    /// P0-NL ACCEPT (fail-closed): requests that the registry SILENTLY
    /// MIS-RESOLVES (returned a confident-wrong, strict-"verified" WRONG program
    /// before this gate) must be REFUSED via ClarificationNeeded — caught by the
    /// EMERGENT STRUCTURAL signals (array-domain-vs-scalar-op mismatch derived from
    /// signatures, operation-not-named), never a phrase blocklist and never a
    /// value-noun / type-noun wordlist. A confidently-wrong coding agent is worse
    /// than one that asks. Empirically:
    ///   * "...sum of squares OF AN ARRAY" → `add` (scalar sig): the array operand
    ///     the request names ("array" resolves to an array-input op) does not match
    ///     the scalar resolved op → DOMAIN MISMATCH → refuse.
    ///   * "parse a CSV file" → no request token names the resolved op at all (the
    ///     content words resolve to nothing above the confidence floor) → refuse.
    #[test]
    fn nl_failclosed_refuses_confident_wrong_resolution() {
        let bridge = LinguigenesisBridge::new();
        for phrase in [
            "return the sum of squares of an array",
            "parse a CSV file",
        ] {
            match bridge.nl_to_requirement(phrase) {
                Err(BridgeError::ClarificationNeeded { .. }) => {}
                Err(BridgeError::ParseError(_)) => {}
                other => panic!(
                    "request {phrase:?} must fail closed (ClarificationNeeded/ParseError), \
                     got {other:?} — confident-wrong resolution leaked"
                ),
            }
        }
    }

    /// DOCUMENTED type-case (Bucket B of the unseen-phrasing benchmark): "reverse a
    /// string" may PASS or REFUSE. The coding registry currently has no `Type`
    /// entity for "string", so — by design, per the EMERGENT-only rule — the bridge
    /// cannot detect a string/Vec type mismatch without a value-type wordlist (which
    /// the global guardrail forbids). It therefore resolves "reverse" to the array
    /// `reverse`. This is acceptable (a benign over-acceptance, not a confidently-
    /// WRONG numeric answer); tightening it requires adding a `Type` entity to
    /// linguigenesis-core (a documented follow-on), NOT a phrase/type blocklist here.
    #[test]
    fn nl_reverse_a_string_is_a_documented_type_case() {
        let bridge = LinguigenesisBridge::new();
        // Whatever the outcome, it must NOT be a non-refusal that ALSO produces a
        // composed-pipeline mis-solve; the type case stays a plain single-op resolve.
        assert!(
            bridge.try_compose_pipeline("reverse a string").is_none(),
            "'reverse a string' must not be mis-built as a numeric pipeline"
        );
        // It is allowed to resolve (single-op reverse) OR refuse; both are fine.
        let _ = bridge.nl_to_requirement("reverse a string");
    }

    /// P0-NL must NOT over-refuse: genuine in-vocab single-op requests whose
    /// content word actually names the resolved op, with no type mismatch, still
    /// resolve to a Requirement (and synthesize). Proves the gate keys on
    /// structural signals, not on "any registry match is suspect".
    #[test]
    fn nl_failclosed_keeps_genuine_in_vocab_ops() {
        let bridge = LinguigenesisBridge::new();
        // Resolve cleanly (guard does not fire).
        for phrase in ["add two numbers", "square a number"] {
            bridge.nl_to_requirement(phrase).unwrap_or_else(|e| {
                panic!("genuine request {phrase:?} must still resolve, got {e:?}")
            });
        }
        // And still synthesize end-to-end.
        let result = bridge
            .synthesize_from_description("add two numbers", Some("add"))
            .expect("genuine op must still synthesize");
        assert!(result.success, "add must still solve: {:?}", result.error);
    }

    /// P0-NL must EXEMPT inline-example requests: the user supplied the spec
    /// directly (comprehension bypassed), so the operation-not-named signal must
    /// not gate them even though the demonstrated op has no registry identity.
    #[test]
    fn nl_failclosed_exempts_inline_example_requests() {
        let bridge = LinguigenesisBridge::new();
        let req = bridge
            .nl_to_requirement("a function mapping [1,2,3] -> [2,3,4] and [5,6] -> [6,7]")
            .expect("inline-example request must be exempt from the fail-closed gate");
        assert_eq!(req.examples.len(), 2);
    }

    #[test]
    fn test_bridge_creation() {
        let bridge = LinguigenesisBridge::new();
        // Should successfully create with default entities
        let registry = bridge.registry.read().unwrap();
        assert!(registry.stats().total_entities > 0);
    }

    #[test]
    fn test_nl_to_examples_add() {
        let bridge = LinguigenesisBridge::new();
        let examples = bridge.nl_to_examples("add two numbers").unwrap();
        assert!(!examples.is_empty());
    }

    #[test]
    fn test_nl_to_examples_map_double() {
        let bridge = LinguigenesisBridge::new();
        let examples = bridge.nl_to_examples("map the array").unwrap();
        assert!(!examples.is_empty());
        assert_eq!(examples[0].expected, Value::int_array(&[2, 4, 6]));
    }


    #[test]
    fn test_synthesize_from_description_add() {
        let bridge = LinguigenesisBridge::new();
        let result = bridge
            .synthesize_from_description("add two numbers", Some("add"))
            .unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_nl_to_examples_reverse_array() {
        let bridge = LinguigenesisBridge::new();
        let examples = bridge.nl_to_examples("reverse array").unwrap();
        assert!(!examples.is_empty());
        assert!(
            matches!(examples[0].expected, Value::Array(_)),
            "expected={:?}",
            examples[0].expected
        );
    }

    #[test]
    fn test_synthesize_from_description_reverse_array() {
        let bridge = LinguigenesisBridge::new();
        let result = bridge
            .synthesize_from_description("reverse array", Some("reverse"))
            .unwrap();
        assert!(result.success, "failed: {:?}", result.error);
        assert!(
            result.method.contains("array_transform")
                || result.method.contains("search_")
                || result.code.contains("push"),
            "method={} code={}",
            result.method,
            result.code
        );
    }

    #[test]
    fn test_get_belief_state() {
        let bridge = LinguigenesisBridge::new();
        let belief = bridge.get_belief_state("reverse the array").unwrap();
        assert_eq!(
            belief.intent.intent_type,
            linguigenesis_core::belief::IntentType::DataTransformation
        );
    }

    #[test]
    fn test_nl_to_examples_combine() {
        let bridge = LinguigenesisBridge::new();
        let examples = bridge.nl_to_examples("combine two numbers").unwrap();
        assert!(!examples.is_empty());
        assert_eq!(examples[0].expected, Value::Int(5));
    }

    #[test]
    fn test_clarification_on_gibberish() {
        let bridge = LinguigenesisBridge::new();
        let err = bridge
            .nl_to_requirement("xyzqwerty qwerty qwerty")
            .unwrap_err();
        assert!(matches!(err, BridgeError::ClarificationNeeded { .. }));
    }

    #[test]
    fn inline_examples_synthesize_unseen_operation() {
        // "quux" is not a registry operation. The agent should still synthesize
        // it purely from the demonstrated I/O examples (here: multiply by 10),
        // routing through the same typed solver as the verified benchmark.
        let bridge = LinguigenesisBridge::new();
        let req = bridge
            .nl_to_requirement("implement quux(1)=10, quux(2)=20, quux(3)=30")
            .expect("inline examples should yield a ready requirement");
        assert_eq!(req.examples.len(), 3, "examples: {:?}", req.examples);
        assert_eq!(req.function_name, "quux");
        let result = bridge
            .synthesize_from_requirement(&req, Some(&req.function_name))
            .expect("synthesis call");
        assert!(result.success, "failed to synthesize unseen op: {:?}", result.error);
    }

    #[test]
    fn inline_examples_array_unseen_operation() {
        let bridge = LinguigenesisBridge::new();
        let req = bridge
            .nl_to_requirement("a function mapping [1,2,3] -> [2,3,4] and [5,6] -> [6,7]")
            .expect("inline array examples ready");
        assert_eq!(req.examples.len(), 2);
        let result = bridge
            .synthesize_from_requirement(&req, Some(&req.function_name))
            .expect("synthesis call");
        assert!(result.success, "failed array synth: {:?}", result.error);
    }

    #[test]
    fn inline_examples_contradiction_asks_for_clarification() {
        // Same input, two different outputs describes no deterministic function.
        // The agent must flag the conflict and ask the user — never silently
        // emit a probabilistic sampler and call it a successful synthesis.
        let bridge = LinguigenesisBridge::new();
        match bridge.nl_to_requirement("define bad(1)=1, bad(1)=2") {
            Err(BridgeError::ClarificationNeeded { partial, questions }) => {
                assert!(
                    partial.unresolved.iter().any(|u| u.starts_with("conflicting examples")),
                    "unresolved={:?}",
                    partial.unresolved
                );
                assert!(!questions.is_empty());
            }
            other => panic!("expected ClarificationNeeded for conflicting examples, got {other:?}"),
        }
    }

    // ---- Newly-registered ops: NL phrase (NO inline examples) must resolve,
    // synthesize, AND generalize. Generalization is proven by executing the
    // synthesized program on HOLDOUT inputs absent from the registry
    // `example_cases`, so a green run rejects example overfit. ----

    fn ex(inputs: Vec<Value>, expected: Value) -> Example {
        Example { inputs, expected }
    }

    /// Resolve `phrase` from NL alone (no inline examples), synthesize, and
    /// verify the synthesized program against holdout inputs.
    fn assert_nl_synthesizes_and_generalizes(
        phrase: &str,
        fn_name: &str,
        signature: &'static str,
        holdouts: Vec<Example>,
    ) {
        let bridge = LinguigenesisBridge::new();
        let result = bridge
            .synthesize_from_description(phrase, None)
            .unwrap_or_else(|e| panic!("{phrase:?}: NL did not resolve/synthesize: {e}"));
        assert!(
            result.success,
            "{phrase:?}: solver returned failure (method={}, err={:?})",
            result.method, result.error
        );
        let holdout_problem = Problem {
            name: fn_name.to_string(),
            category: "test",
            description: "holdout generalization",
            signature,
            examples: holdouts,
            ..Default::default()
        };
        crate::runtime::verify_problem_code(&holdout_problem, &result.code).unwrap_or_else(|e| {
            panic!(
                "{phrase:?}: OVERFIT — holdout generalization failed: {e}\nmethod={}\nCODE:\n{}",
                result.method, result.code
            )
        });
    }

    #[test]
    fn nl_abs_synthesizes_and_generalizes() {
        assert_nl_synthesizes_and_generalizes(
            "compute the absolute value of a number",
            "abs",
            "fn abs(a: i64) -> i64",
            vec![
                ex(vec![Value::Int(-50)], Value::Int(50)),
                ex(vec![Value::Int(42)], Value::Int(42)),
                ex(vec![Value::Int(-100)], Value::Int(100)),
                ex(vec![Value::Int(7)], Value::Int(7)),
                ex(vec![Value::Int(-6)], Value::Int(6)),
            ],
        );
    }

    #[test]
    fn nl_triple_synthesizes_and_generalizes() {
        assert_nl_synthesizes_and_generalizes(
            "triple a number",
            "triple",
            "fn triple(a: i64) -> i64",
            vec![
                ex(vec![Value::Int(7)], Value::Int(21)),
                ex(vec![Value::Int(100)], Value::Int(300)),
                ex(vec![Value::Int(-50)], Value::Int(-150)),
                ex(vec![Value::Int(8)], Value::Int(24)),
            ],
        );
    }

    #[test]
    fn nl_square_synthesizes_and_generalizes() {
        assert_nl_synthesizes_and_generalizes(
            "square a number",
            "square",
            "fn square(a: i64) -> i64",
            vec![
                ex(vec![Value::Int(7)], Value::Int(49)),
                ex(vec![Value::Int(10)], Value::Int(100)),
                ex(vec![Value::Int(1)], Value::Int(1)),
                ex(vec![Value::Int(9)], Value::Int(81)),
            ],
        );
    }

    #[test]
    fn nl_negate_synthesizes_and_generalizes() {
        assert_nl_synthesizes_and_generalizes(
            "negate a number",
            "negate",
            "fn negate(a: i64) -> i64",
            vec![
                ex(vec![Value::Int(100)], Value::Int(-100)),
                ex(vec![Value::Int(-50)], Value::Int(50)),
                ex(vec![Value::Int(13)], Value::Int(-13)),
            ],
        );
    }

    #[test]
    fn nl_array_sum_synthesizes_and_generalizes() {
        assert_nl_synthesizes_and_generalizes(
            "compute the total of an array",
            "array_sum",
            "fn array_sum(a: [i64]) -> i64",
            vec![
                ex(vec![Value::int_array(&[100, 1])], Value::Int(101)),
                ex(vec![Value::int_array(&[4, 4, 4])], Value::Int(12)),
                ex(vec![Value::int_array(&[-1, -2, -3])], Value::Int(-6)),
                ex(vec![Value::int_array(&[5])], Value::Int(5)),
            ],
        );
    }

    #[test]
    fn nl_array_max_synthesizes_and_generalizes() {
        assert_nl_synthesizes_and_generalizes(
            "compute the largest of an array",
            "array_max",
            "fn array_max(a: [i64]) -> i64",
            vec![
                ex(vec![Value::int_array(&[100, 2, 50])], Value::Int(100)),
                ex(vec![Value::int_array(&[5, 5, 5])], Value::Int(5)),
                ex(vec![Value::int_array(&[1, 2, 3, 4])], Value::Int(4)),
            ],
        );
    }

    #[test]
    fn nl_array_min_synthesizes_and_generalizes() {
        assert_nl_synthesizes_and_generalizes(
            "compute the smallest of an array",
            "array_min",
            "fn array_min(a: [i64]) -> i64",
            vec![
                ex(vec![Value::int_array(&[100, 2, 50])], Value::Int(2)),
                ex(vec![Value::int_array(&[5, 5, 5])], Value::Int(5)),
                ex(vec![Value::int_array(&[4, 3, 2, 1])], Value::Int(1)),
            ],
        );
    }

    /// `sum3` is reachable as a proven registry op even though plain English has
    /// no single-token trigger distinct from 2-arg `add` (see roadmap deferral
    /// note). Drive it through the registry-seed requirement path and prove it
    /// generalizes on holdouts.
    #[test]
    fn registry_sum3_synthesizes_and_generalizes() {
        use linguigenesis_core::entity::EntityType;
        let bridge = LinguigenesisBridge::new();
        let registry = bridge.registry_clone().expect("registry clone");
        let entity = registry
            .get_by_type(&EntityType::Function)
            .into_iter()
            .find(|e| e.lemma == "sum3")
            .expect("sum3 registered");
        let req = SynthesisRequirement::from_operation_entity(&entity).expect("sum3 requirement");
        let result = bridge
            .synthesize_from_requirement(&req, Some(&req.function_name))
            .expect("sum3 synthesis");
        assert!(result.success, "sum3 failed: {:?}", result.error);
        let holdout_problem = Problem {
            name: "sum3".to_string(),
            category: "test",
            description: "holdout",
            signature: "fn sum3(a: i64, b: i64, c: i64) -> i64",
            examples: vec![
                ex(vec![Value::Int(5), Value::Int(5), Value::Int(5)], Value::Int(15)),
                ex(vec![Value::Int(10), Value::Int(-3), Value::Int(1)], Value::Int(8)),
                ex(vec![Value::Int(0), Value::Int(0), Value::Int(7)], Value::Int(7)),
                ex(vec![Value::Int(100), Value::Int(1), Value::Int(1)], Value::Int(102)),
            ],
            ..Default::default()
        };
        crate::runtime::verify_problem_code(&holdout_problem, &result.code)
            .unwrap_or_else(|e| panic!("sum3 OVERFIT: {e}\nCODE:\n{}", result.code));
    }

    /// Integrity gate: every operation declared in the coding registry (i.e.
    /// every function entity carrying `example_cases`) must actually synthesize
    /// through the real solver. This prevents the registry from advertising
    /// vocabulary the engine cannot deliver — declared capability == proven
    /// capability, per the no-cheating contract.
    #[test]
    fn every_registry_operation_is_synthesizable() {
        use linguigenesis_core::entity::EntityType;

        let bridge = LinguigenesisBridge::new();
        let registry = bridge.registry_clone().expect("registry clone");

        let mut checked = 0usize;
        let mut failures: Vec<String> = Vec::new();
        for entity in registry.get_by_type(&EntityType::Function) {
            let req = match SynthesisRequirement::from_operation_entity(&entity) {
                Some(r) => r,
                None => continue,
            };
            checked += 1;
            match bridge.synthesize_from_requirement(&req, Some(&req.function_name)) {
                Ok(result) if result.success => {}
                Ok(result) => failures.push(format!(
                    "{}: solver returned failure (method={}, err={:?})",
                    entity.lemma, result.method, result.error
                )),
                Err(e) => failures.push(format!("{}: {}", entity.lemma, e)),
            }
        }

        assert!(checked >= 20, "expected >=20 registry ops, checked {checked}");
        assert!(
            failures.is_empty(),
            "{}/{} registry ops failed to synthesize:\n{}",
            failures.len(),
            checked,
            failures.join("\n")
        );
    }

    /// Build a `SynthesisRequirement` directly from a registry op by name (no NL),
    /// so a soundness test can target a SPECIFIC op's spec.
    fn req_for_op(bridge: &LinguigenesisBridge, fn_name: &str) -> SynthesisRequirement {
        use linguigenesis_core::entity::EntityType;
        let registry = bridge.registry_clone().expect("registry clone");
        let entity = registry
            .get_by_type(&EntityType::Function)
            .into_iter()
            .find(|e| {
                e.get_property("default_fn_name").map(|f| f == fn_name).unwrap_or(false)
                    || e.lemma == fn_name
            })
            .unwrap_or_else(|| panic!("registry op {fn_name:?} not found"));
        SynthesisRequirement::from_operation_entity(&entity)
            .unwrap_or_else(|| panic!("op {fn_name:?} has no synthesizable spec"))
    }

    /// HARDEN-2 FIX B ACCEPT (un-gameable, single-op overfit floor). A SCALAR
    /// registry op carrying <2 DISTINCT example rows is verified examples-only by
    /// `verify_problem_code_strict` (no runnable reference → holdouts degrade to
    /// the seed rows), so ANY program reproducing that one pair "passes" — a
    /// confident-WRONG overfit. The fail-closed gate must now REFUSE it.
    ///
    /// The test is differential and cannot be gamed:
    ///   * PRIOR PATH PROOF: the op DOES synthesize a "successful" program through
    ///     the un-gated `synthesize_from_requirement` (the exact confident-wrong
    ///     emission this fix prevents);
    ///   * NEW GATE: `fail_closed_reason` (HARD category) REFUSES it, and the
    ///     reason is the EMERGENT example-floor signal (not a phrase list);
    ///   * CONTRAST: a genuine scalar op with >=2 distinct rows (`negate`) is NOT
    ///     refused, proving the floor discriminates on evidence, not over-refuses.
    #[test]
    fn failclosed_floor_refuses_thin_scalar_single_op() {
        let bridge = LinguigenesisBridge::new();

        // `subtract` ships exactly one example row and is scalar (i64,i64 -> i64).
        let thin = req_for_op(&bridge, "subtract");
        let distinct: std::collections::BTreeSet<_> =
            thin.examples.iter().map(|e| format!("{e:?}")).collect();
        assert!(
            distinct.len() < 2,
            "fixture invalid: 'subtract' must carry <2 distinct rows for this floor test, got {}",
            distinct.len()
        );

        // PRIOR PATH: the un-gated solver WOULD emit a "successful" program — this
        // is the confident-wrong emission the floor must stop.
        let prior = bridge
            .synthesize_from_requirement(&thin, Some(&thin.function_name))
            .expect("subtract synthesis runs");
        assert!(
            prior.success,
            "fixture invalid: the un-gated path must succeed (else there is nothing to gate)"
        );

        // NEW GATE: must REFUSE, via the emergent example-floor signal.
        let reason = bridge
            .fail_closed_reason("subtract two numbers", &thin)
            .expect("thin scalar single-op must be refused by the floor");
        assert!(
            reason.contains("distinct example row"),
            "refusal must come from the emergent example-floor signal, got: {reason}"
        );

        // CONTRAST: a genuine multi-example scalar op is NOT refused by the floor.
        let genuine = req_for_op(&bridge, "negate");
        assert!(
            bridge.fail_closed_reason("negate a number", &genuine).is_none(),
            "the floor must NOT over-refuse a genuine >=2-example scalar op (negate)"
        );
    }

    /// HARDEN-2 FIX B ACCEPT (single-op FRESH holdout). When a registry op carries
    /// >=3 distinct rows, `problem_from_requirement` reserves one as a HELD-OUT
    /// generalization probe (NOT shown to the solver) so strict verification is
    /// differential, not examples-only. Prove the reserved holdout is real and the
    /// synthesized program passes it (so a memoriser would be caught).
    #[test]
    fn single_op_problem_reserves_fresh_holdout_and_strict_verifies() {
        let bridge = LinguigenesisBridge::new();
        let req = req_for_op(&bridge, "negate"); // 5 distinct rows -> reserve 1
        let problem = bridge
            .problem_from_requirement(&req, Some(&req.function_name))
            .expect("negate problem");
        assert!(
            !problem.holdouts.is_empty(),
            "a >=3-example single-op problem must reserve a FRESH (held-out) row"
        );
        // The reserved holdout must be a row the solver never saw as a seed.
        for h in &problem.holdouts {
            assert!(
                !problem.examples.contains(h),
                "reserved holdout {h:?} leaked into the seed examples (not differential)"
            );
        }
        // Synthesize from the (reduced) seed and STRICT-verify against the fresh
        // held-out row — a memoriser of the seeds would fail here.
        let solved = crate::solver::solve_problem(&problem);
        assert!(solved.success, "negate must synthesize from reduced seed: {:?}", solved.error);
        crate::runtime::verify_problem_code_strict(&problem, &solved.code)
            .unwrap_or_else(|e| panic!("negate failed FRESH-holdout strict verify: {e}"));
    }

    /// HARDEN-2 FIX A ACCEPT (portability): the registry resolves through a
    /// COMPILE-TIME absolute base (`CARGO_MANIFEST_DIR`-relative), so it loads
    /// regardless of CWD / $HOME and the explicit load-error surface reports
    /// HEALTHY. We assert the COMPILE-TIME base is an absolute path that points
    /// at the real data file (this is what makes load cwd/HOME-independent), and
    /// that a freshly-built bridge reports no load error + can synthesize an
    /// in-vocab op. (Full cross-CWD/HOME proof is the CLI transcript battery —
    /// mutating process-global CWD/HOME inside a parallel test binary would race
    /// other tests, so the un-gameable location proof lives in the CLI accept run.)
    #[test]
    fn registry_resolves_via_compile_time_absolute_base() {
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../linguigenesis/data/coding_registry.json");
        assert!(
            base.is_absolute(),
            "compile-time base must be absolute (cwd/HOME-independent), got {base:?}"
        );
        assert!(
            base.exists(),
            "compile-time base must point at the real coding_registry.json: {base:?}"
        );
        let bridge = LinguigenesisBridge::new();
        assert!(
            bridge.registry_load_error().is_none(),
            "registry must report healthy (operations available) when loaded"
        );
        let r = bridge.synthesize_from_description("add two numbers", Some("add"));
        assert!(r.is_ok() && r.unwrap().success, "add must synthesize with the loaded registry");
    }

    // ===== UNWALL-2-CALLNODE-NL: B-calls-A from an English request =====

    fn unwall2_fresh(tag: &str) -> std::path::PathBuf {
        let root = std::env::temp_dir().join(format!(
            "nsynth_unwall2_{tag}_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    /// ACCEPT (un-gameable, CLI-equivalent end-to-end): an English 'B uses/calls A'
    /// request flows through the REAL NL door (`synthesize_project`: comprehend →
    /// emergent dep-detection → COMPOSED-example derivation → Call-node search) and
    /// yields a 2-module crate where the CONSUMER genuinely CALLS the producer
    /// (a real call naming `square`, NOT an inlined re-derivation), the crate
    /// COMPILES through the cargo-check writer gate, and a GENERATED assert passes
    /// via `cargo test`. The composed consumer examples are DERIVED by RUNNING the
    /// solved producer + residual inside `synthesize_consumer_with_call` — never
    /// fabricated.
    #[test]
    fn unwall2_consumer_calls_producer_endtoend() {
        let bridge = LinguigenesisBridge::new();
        assert!(bridge.registry_load_error().is_none(), "registry must load");

        // "increments its square using square" → B(x) = square(x) + 1.
        let request = "a function that squares a number \
                       and a function that increments its square using square";
        let (solved, skipped) = bridge
            .synthesize_project(request)
            .expect("synthesize_project must run");
        eprintln!("[UNWALL2] solved={:?} skipped={:?}",
            solved.iter().map(|(n, r)| (n.clone(), r.method.clone(), r.code.clone())).collect::<Vec<_>>(),
            skipped);

        assert_eq!(solved.len(), 2, "both producer + consumer must solve; skipped={skipped:?}");
        // Producer "square" present.
        let producer = solved.iter().find(|(n, _)| n == "square").expect("square solved");
        assert!(producer.1.code.contains("fn square"), "producer is square: {}", producer.1.code);
        // Consumer "increment" present AND genuinely CALLS square (not inlined).
        let consumer = solved.iter().find(|(n, _)| n == "increment").expect("increment solved");
        assert!(
            crate::agent::repo::body_calls_fn(&consumer.1.code, "square"),
            "consumer must CALL square (not inline a*a): {}",
            consumer.1.code
        );
        // NOT inlined: the body must NOT contain a raw `a * a` self-multiplication.
        assert!(
            !consumer.1.code.replace(' ', "").contains("a*a"),
            "consumer must NOT inline a*a — it must reuse square: {}",
            consumer.1.code
        );

        // END-TO-END: write the 2-module crate via the real writer (use-injection +
        // cargo-check gate), then append a generated assert and run cargo test.
        let components: Vec<(String, String)> =
            solved.iter().map(|(n, r)| (n.clone(), r.code.clone())).collect();
        let root = unwall2_fresh("square_inc");
        let outcome = crate::agent::repo::write_synthesized_project(&root, "square_inc", &components)
            .expect("write crate");
        assert!(
            outcome.compile.is_ok(),
            "2-module crate must compile clean: {:?}",
            outcome.compile
        );
        // The consumer module must carry the injected `use crate::square::square;`.
        let cons_mod = std::fs::read_to_string(root.join("src/increment.rs")).unwrap();
        assert!(
            cons_mod.contains("use crate::square::square;"),
            "consumer module must import square: {cons_mod}"
        );
        assert!(
            cons_mod.contains("square("),
            "consumer module body must call square(...): {cons_mod}"
        );

        // GENERATED assert: increment(3) must equal 3*3 + 1 == 10.
        let mut lib = std::fs::read_to_string(root.join("src/lib.rs")).unwrap();
        lib.push_str(
            "\n#[cfg(test)]\nmod unwall2_tests {\n    use super::*;\n    #[test]\n    fn consumer_calls_producer() {\n        assert_eq!(increment(3), 10);\n        assert_eq!(increment(4), 17);\n    }\n}\n",
        );
        std::fs::write(root.join("src/lib.rs"), &lib).unwrap();
        let runtime = crate::agent::tools::SecureToolRuntime::for_repo_repair(
            root.clone(),
            crate::agent::repo::GuardrailPolicy::default(),
        );
        let test_run = runtime
            .run_verification_command("cargo test")
            .expect("cargo test must run");
        assert!(
            test_run.success,
            "generated assert must pass:\nstdout:\n{}\nstderr:\n{}\nconsumer:\n{cons_mod}",
            test_run.stdout, test_run.stderr
        );
        eprintln!("[UNWALL2] consumer module:\n{cons_mod}");
        eprintln!("[UNWALL2] cargo test stdout:\n{}", test_run.stdout);
        if std::env::var("NSYNTH_KEEP_CRATE").is_err() {
            let _ = std::fs::remove_dir_all(root);
        }
    }

    /// DIFFERENTIAL: an INDEPENDENT-sibling request ('negate' + 'triple', no
    /// call/use cue) must yield TWO INDEPENDENT functions with NO spurious call —
    /// `deps` is empty so neither body references the other. This is the
    /// no-false-positive guard for the dep path.
    #[test]
    fn unwall2_independent_siblings_have_no_call() {
        let bridge = LinguigenesisBridge::new();
        let request = "a module with a function that negates a number \
                       and a function that triples a number";
        let (solved, skipped) = bridge
            .synthesize_project(request)
            .expect("synthesize_project must run");
        assert_eq!(solved.len(), 2, "both independent fns solve; skipped={skipped:?}");
        for (name, res) in &solved {
            // Neither sibling may call the other (independent → no Call discovered).
            for (other, _) in &solved {
                if other == name {
                    continue;
                }
                assert!(
                    !crate::agent::repo::body_calls_fn(&res.code, other),
                    "independent sibling '{name}' must NOT call '{other}': {}",
                    res.code
                );
            }
        }
    }

    /// REGRESSION: a SINGLE-function request is a 1-component plan (deps empty), so
    /// the dep path is never taken and behaviour is unchanged (the existing
    /// single-op door still solves it).
    #[test]
    fn unwall2_single_function_unchanged() {
        let bridge = LinguigenesisBridge::new();
        let (solved, _skipped) = bridge
            .synthesize_project("a function that squares a number")
            .expect("synthesize_project must run");
        assert_eq!(solved.len(), 1, "single-function request yields 1 component");
        assert!(solved[0].1.code.contains("fn square"), "single fn is square: {}", solved[0].1.code);
    }
}
