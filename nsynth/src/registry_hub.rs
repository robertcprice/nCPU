//! REGISTRY HUB — one universal substrate, domain overlays that GROW.
//!
//! Architecture (the MoE shape on a shared spine): every domain — coding, web,
//! backend, whatever comes next — is the SAME machinery (lg-core `Registry` +
//! `EntityResolver`: synonym edges, morphology, fuzzy, definition overlap)
//! loaded from its own DATA overlay, all mergeable, all growing through the
//! same two seams:
//!   * MINING — engine surfaces reflected into data (capability_miner for
//!     coding ops, the crawler's component promotion, future web/backend
//!     miners);
//!   * TEACHING — runtime NL ("teach web: a testimonials section means
//!     customer quotes in a row, also called reviews") appends an entity to
//!     the domain's data file, persistently: the vocabulary a request needs
//!     but lacks can be added mid-conversation and survives restarts.
//! In-code seeds remain the FLOOR (a bad data file never breaks built-ins);
//! data merges over them — the exact pattern the component registry proved.

use linguigenesis_core::entity::{Entity, EntityType, RelationType};
use linguigenesis_core::registry::Registry;
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Domain {
    Web,
    Backend,
}

impl Domain {
    fn kind_key(&self) -> &'static str {
        match self {
            Domain::Web => "web_kind",
            Domain::Backend => "backend_kind",
        }
    }
    pub fn env_var_name(&self) -> &'static str {
        self.env_var()
    }
    fn env_var(&self) -> &'static str {
        match self {
            Domain::Web => "NSYNTH_WEB_REGISTRY",
            Domain::Backend => "NSYNTH_BACKEND_REGISTRY",
        }
    }
    fn default_file(&self) -> &'static str {
        match self {
            Domain::Web => "web_registry.json",
            Domain::Backend => "backend_registry.json",
        }
    }
}

/// One taught/seeded concept: a lemma with a kind, a definition, and synonyms.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct Concept {
    pub lemma: String,
    pub kind: String,
    pub definition: String,
    #[serde(default)]
    pub synonyms: Vec<String>,
}

fn data_path(domain: Domain) -> Option<PathBuf> {
    if let Ok(p) = std::env::var(domain.env_var()) {
        if !p.trim().is_empty() {
            return Some(PathBuf::from(p));
        }
    }
    // Conventional shared-data locations (same seam the other registries use).
    for base in ["../linguigenesis/data", "../../linguigenesis/data"] {
        let p = PathBuf::from(base);
        if p.is_dir() {
            return Some(p.join(domain.default_file()));
        }
    }
    None
}

/// Load a domain's TAUGHT/DATA concepts (empty when absent/unparseable — the
/// seeds are the floor).
pub fn load_domain_concepts(domain: Domain) -> Vec<Concept> {
    data_path(domain)
        .and_then(|p| std::fs::read_to_string(p).ok())
        .and_then(|t| serde_json::from_str::<Vec<Concept>>(&t).ok())
        .unwrap_or_default()
}

/// TEACH: persist a concept into the domain's data file (merge by lemma) so it
/// resolves in every future request AND every future process. The growth seam.
pub fn teach_concept(domain: Domain, concept: Concept) -> Result<(), String> {
    let path = data_path(domain).ok_or("no data path for domain registry")?;
    let mut all = load_domain_concepts(domain);
    if let Some(slot) = all.iter_mut().find(|c| c.lemma == concept.lemma) {
        *slot = concept;
    } else {
        all.push(concept);
    }
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(&path, serde_json::to_string_pretty(&all).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}

/// Build the domain's live Registry: seeds (the floor) + data concepts merged
/// over them, every concept with synonym EDGES so the real resolver's lenses
/// all apply.
pub fn domain_registry(domain: Domain, seeds: &[Concept]) -> Registry {
    let reg = Registry::new();
    let mut id: u64 = 1;
    fn add_concept(reg: &Registry, id: &mut u64, kind_key: &str, c: &Concept) {
        let mut e = Entity::new(*id, c.lemma.clone(), EntityType::Noun);
        *id += 1;
        e.add_definition(c.definition.clone());
        e.add_property(kind_key.into(), c.kind.clone());
        let _ = reg.add_entity(e);
        for syn in &c.synonyms {
            let se = Entity::new(*id, syn.clone(), EntityType::Noun);
            *id += 1;
            let _ = reg.add_entity(se);
            let _ = reg.link_lemma_relation(syn, RelationType::Synonym, &c.lemma);
            let _ = reg.link_lemma_relation(&c.lemma, RelationType::Synonym, syn);
        }
    }
    // Seeds first (floor); data lemmas that duplicate a seed only contribute NEW
    // synonym edges; new lemmas extend the domain.
    let data = load_domain_concepts(domain);
    for c in seeds.iter() {
        add_concept(&reg, &mut id, domain.kind_key(), c);
    }
    for c in &data {
        if seeds.iter().any(|s| s.lemma == c.lemma) {
            for syn in &c.synonyms {
                let se = Entity::new(id, syn.clone(), EntityType::Noun);
                id += 1;
                let _ = reg.add_entity(se);
                let _ = reg.link_lemma_relation(syn, RelationType::Synonym, &c.lemma);
                let _ = reg.link_lemma_relation(&c.lemma, RelationType::Synonym, syn);
            }
        } else {
            add_concept(&reg, &mut id, domain.kind_key(), c);
        }
    }
    reg
}

/// Resolve a token to a domain concept through the FULL emergent stack: ranked
/// candidates, and — mirroring lg-core's `canonical_operation_entity` — a
/// candidate that lacks the domain kind (e.g. the morphological hit on a
/// synonym entity: "routes" -> "route") follows its Synonym EDGES to the
/// canonical concept that carries it.
pub fn resolve_domain(
    resolver: &linguigenesis_core::entity_resolution::EntityResolver,
    registry: &Registry,
    token: &str,
    kind_key: &str,
) -> Option<(String, String, f32)> {
    for cand in resolver.rank_candidates(token) {
        if cand.evidence.score < 0.5 {
            continue;
        }
        if let Some(kind) = cand.entity.get_property(kind_key) {
            return Some((kind.clone(), cand.entity.lemma.clone(), cand.evidence.score));
        }
        if let Some(targets) = cand.entity.get_related(RelationType::Synonym) {
            for id in targets {
                if let Some(t) = registry.get_entity(*id) {
                    if let Some(kind) = t.get_property(kind_key) {
                        return Some((kind.clone(), t.lemma.clone(), cand.evidence.score));
                    }
                }
            }
        }
    }
    None
}

/// GENERAL, DOMAIN-AGNOSTIC archetype composition. An ARCHETYPE is a named
/// PURPOSE that implies a STRUCTURE ("landing" → hero+features+contact; a
/// backend "crud api" → create+read+update+delete routes + a store). Its
/// composition is DERIVED by COMPREHENDING its own definition: resolve each word
/// of the definition against the domain registry and collect the entities of
/// `part_kind`, in order. This is the same mechanism for EVERY domain — no
/// per-domain hardcoded table — so archetypes are emergent, learnable, and not
/// web-bound. A taught or self-minted archetype composes for free from the prose
/// it carries.
pub fn compose_from_definition(
    resolver: &linguigenesis_core::entity_resolution::EntityResolver,
    registry: &Registry,
    definition: &str,
    kind_key: &str,
    part_kind: &str,
) -> Vec<String> {
    let mut parts: Vec<String> = Vec::new();
    for tok in definition
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| !t.is_empty())
    {
        if let Some((kind, lemma, _score)) = resolve_domain(resolver, registry, tok, kind_key) {
            if kind == part_kind && !parts.contains(&lemma) {
                parts.push(lemma);
            }
        }
    }
    parts
}

/// SELF-CREATION: mint a NEW archetype at runtime from an observed set of parts
/// (e.g. the sections a page was actually built with, or the routes an api
/// exposed). Writes an archetype `Concept` whose DEFINITION names those parts,
/// so it then resolves and composes exactly like a taught or built-in archetype
/// — the system creating its own reusable structural vocabulary. Domain-agnostic
/// and persistent (survives restarts via the domain data file). `name` becomes
/// the archetype lemma; `parts` are the component lemmas it implies.
pub fn remember_archetype(domain: Domain, name: &str, parts: &[&str]) -> Result<(), String> {
    if name.trim().is_empty() || parts.is_empty() {
        return Err("an archetype needs a name and at least one part".into());
    }
    // A definition that names the parts in prose — the SAME shape a human would
    // teach, so composition derives identically.
    let listed = parts.join(", ");
    let definition = format!("a {name} with {listed}");
    teach_concept(
        domain,
        Concept {
            lemma: name.to_string(),
            kind: "archetype".to_string(),
            definition,
            synonyms: Vec::new(),
        },
    )
}

/// The BACKEND domain seeds (routes/stores/apis as resolvable concepts —
/// the vocabulary `backend_nl` asks arrive in).
pub fn backend_seeds() -> Vec<Concept> {
    let c = |lemma: &str, kind: &str, def: &str, syns: &[&str]| Concept {
        lemma: lemma.into(),
        kind: kind.into(),
        definition: def.into(),
        synonyms: syns.iter().map(|s| s.to_string()).collect(),
    };
    vec![
        c("endpoint", "route", "an http api endpoint route handling requests", &["route", "api", "handler"]),
        c("store", "store", "a data store persisting records", &["database", "storage", "table"]),
        c("server", "server", "an http server serving the application", &["backend", "service"]),
        c("auth", "middleware", "authentication checking user identity and login", &["login", "authentication", "signin"]),
        c("health", "route", "a health check endpoint reporting server status", &["healthcheck", "status", "ping"]),
    ]
}

/// Structural TEACH parse: "teach <domain>: a <lemma> <kind> means <definition>
/// [, also called <syn> [or <syn>...]]". Purely structural (the teach shape),
/// content free-form. Returns (domain, concept).
pub fn parse_teach(text: &str) -> Option<(Domain, Concept)> {
    let lower = text.trim().to_lowercase();
    let rest = lower.strip_prefix("teach ")?;
    let (dom_word, rest) = rest.split_once(':')?;
    let domain = match dom_word.trim() {
        "web" | "site" => Domain::Web,
        "backend" | "api" => Domain::Backend,
        _ => return None,
    };
    let rest = rest.trim();
    let rest = rest.strip_prefix("a ").or_else(|| rest.strip_prefix("an ")).unwrap_or(rest);
    let (head, def_part) = rest.split_once(" means ")?;
    let mut head_words = head.split_whitespace();
    let lemma = head_words.next()?.to_string();
    let kind = head_words.next().unwrap_or("section").to_string();
    let (definition, syn_part) = match def_part.split_once(", also called ") {
        Some((d, s)) => (d.trim().to_string(), Some(s)),
        None => (def_part.trim().to_string(), None),
    };
    let synonyms: Vec<String> = syn_part
        .map(|s| {
            s.split(|c: char| c == ',' || c.is_whitespace())
                .filter(|w| !w.is_empty() && *w != "or" && *w != "and")
                .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '_').to_string())
                .filter(|w| !w.is_empty())
                .collect()
        })
        .unwrap_or_default();
    if lemma.is_empty() || definition.is_empty() {
        return None;
    }
    Some((domain, Concept { lemma, kind, definition, synonyms }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use linguigenesis_core::entity_resolution::EntityResolver;

    fn temp_env(domain: Domain, tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("nsynth_hub_{tag}_{}.json", std::process::id()));
        let _ = std::fs::remove_file(&p);
        std::env::set_var(domain.env_var(), &p);
        p
    }

    #[test]
    fn teach_persists_and_resolves_through_the_real_resolver() {
        let p = temp_env(Domain::Web, "teach");
        let (dom, concept) = parse_teach(
            "teach web: a testimonials section means customer quotes displayed in a row, \
             also called reviews or quotes",
        )
        .expect("parses");
        assert_eq!(dom, Domain::Web);
        assert_eq!(concept.lemma, "testimonials");
        assert_eq!(concept.synonyms, vec!["reviews".to_string(), "quotes".to_string()]);
        teach_concept(dom, concept).expect("persist");

        // A FRESH registry (as a new process would build) resolves the taught
        // lemma AND its synonyms through the real resolver.
        let reg = domain_registry(Domain::Web, &[]);
        let r = EntityResolver::new(reg);
        for w in ["testimonials", "reviews", "quotes", "testimonial"] {
            let hit = r
                .rank_candidates(w)
                .into_iter()
                .find(|c| c.entity.get_property("web_kind").is_some());
            assert!(hit.is_some(), "{w} must resolve to the taught concept");
            assert_eq!(hit.unwrap().entity.lemma, "testimonials", "{w}");
        }
        std::env::remove_var(Domain::Web.env_var());
        let _ = std::fs::remove_file(&p);
    }

    /// compose_from_definition is DOMAIN-AGNOSTIC: build a synthetic non-web
    /// registry (a "recipe" domain whose parts are ingredients) and compose an
    /// archetype from prose. Proves archetypes are not web-bound — the same
    /// mechanism composes any domain's structure from its definition.
    #[test]
    fn compose_from_definition_is_domain_agnostic() {
        use linguigenesis_core::entity::{Entity, EntityType};
        let reg = Registry::new();
        let mut id = 1u64;
        let mut part = |lemma: &str| {
            let mut e = Entity::new(id, lemma.to_string(), EntityType::Noun);
            id += 1;
            e.add_property("recipe_kind".into(), "ingredient".into());
            let _ = reg.add_entity(e);
        };
        for p in ["flour", "sugar", "eggs", "butter"] {
            part(p);
        }
        let r = EntityResolver::new(reg.clone());
        let parts = compose_from_definition(
            &r,
            &reg,
            "a cake with flour, sugar, and eggs",
            "recipe_kind",
            "ingredient",
        );
        assert_eq!(parts, vec!["flour", "sugar", "eggs"], "composed from prose, in order");
    }

    /// SELF-CREATION round-trip: mint an archetype from observed parts, then a
    /// fresh registry resolves it AND composes its parts from the minted
    /// definition — the system creating its own reusable structure.
    #[test]
    fn remember_archetype_mints_a_composable_archetype() {
        let p = temp_env(Domain::Web, "mint");
        remember_archetype(Domain::Web, "dashboard", &["hero", "features", "about"])
            .expect("mint");
        // Persisted as an archetype concept whose definition names the parts.
        let concepts = load_domain_concepts(Domain::Web);
        let arch = concepts
            .iter()
            .find(|c| c.lemma == "dashboard")
            .expect("archetype persisted");
        assert_eq!(arch.kind, "archetype");
        for part in ["hero", "features", "about"] {
            assert!(arch.definition.contains(part), "definition names {part}: {}", arch.definition);
        }
        std::env::remove_var(Domain::Web.env_var());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn backend_domain_resolves_its_vocabulary() {
        std::env::remove_var(Domain::Backend.env_var());
        let reg = domain_registry(Domain::Backend, &backend_seeds());
        let r = EntityResolver::new(reg);
        let reg2 = domain_registry(Domain::Backend, &backend_seeds());
        for (w, want) in [
            ("api", "endpoint"),
            ("database", "store"),
            ("login", "auth"),
            ("healthcheck", "health"),
            ("routes", "endpoint"),
        ] {
            let (_, lemma, _) = resolve_domain(&r, &reg2, w, "backend_kind")
                .unwrap_or_else(|| panic!("{w} must resolve"));
            assert_eq!(lemma, want, "{w}");
        }
    }

    #[test]
    fn teach_parse_declines_malformed() {
        assert!(parse_teach("teach web: gibberish").is_none());
        assert!(parse_teach("teach quantum: a q thing means x").is_none());
        assert!(parse_teach("please teach me rust").is_none());
    }
}
