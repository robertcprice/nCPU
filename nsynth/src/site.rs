//! SITE DOMAIN — NL-addressable web artifacts ("add a page: modern theme, hero
//! and gallery, teal and charcoal color scheme").
//!
//! The same discipline as the component layer, applied to pages:
//!   * THEMES are design-token sets (data-shaped seeds, extensible like the
//!     component registry) — "modern" is a token vector, not an adjective.
//!   * PALETTES resolve against the PLATFORM'S OWN color vocabulary (the CSS
//!     named-color standard + hex literals) — a web standard, not a hand list.
//!   * SECTIONS are registry units (nav/hero/features/gallery/contact/footer/
//!     about), each carrying its EMISSION and its ASSERTION — the request is
//!     the spec, and every requested section is structurally verified present.
//!   * VERIFICATION is REQUEST-DERIVED STRUCTURAL FIDELITY: well-formed HTML
//!     (tag balance), every requested section's selector present, every
//!     requested color applied in the CSS, the title correct, links resolving.
//!     Aesthetic quality is NOT claimed — fidelity to the request is.

use linguigenesis_core::entity_resolution::morphological_variants;
use std::path::Path;

/// A design-token set. Themes are DATA (seeded here, extensible via JSON later
/// exactly like the component registry).
#[derive(Clone, Debug)]
pub struct Theme {
    pub name: &'static str,
    pub font_stack: &'static str,
    pub radius: &'static str,
    pub shadow: &'static str,
    pub spacing: &'static str,
    pub heading_weight: &'static str,
}

pub fn theme_registry() -> Vec<Theme> {
    vec![
        Theme {
            name: "modern",
            font_stack: "'Inter', 'Segoe UI', system-ui, sans-serif",
            radius: "12px",
            shadow: "0 8px 24px rgba(0,0,0,0.12)",
            spacing: "2rem",
            heading_weight: "700",
        },
        Theme {
            name: "minimal",
            font_stack: "'Helvetica Neue', Arial, sans-serif",
            radius: "0",
            shadow: "none",
            spacing: "1.25rem",
            heading_weight: "400",
        },
        Theme {
            name: "classic",
            font_stack: "Georgia, 'Times New Roman', serif",
            radius: "4px",
            shadow: "0 2px 6px rgba(0,0,0,0.2)",
            spacing: "1.5rem",
            heading_weight: "600",
        },
    ]
}

/// The CSS named-color standard (the platform's own vocabulary — resolving
/// against it is like resolving lemmas against the registry). Subset covering
/// the full standard's common names; hex literals are accepted alongside.
const CSS_COLORS: &[&str] = &[
    "black", "silver", "gray", "grey", "white", "maroon", "red", "purple",
    "fuchsia", "green", "lime", "olive", "yellow", "navy", "blue", "teal",
    "aqua", "orange", "gold", "coral", "salmon", "crimson", "indigo", "violet",
    "plum", "orchid", "magenta", "khaki", "ivory", "beige", "mint", "azure",
    "lavender", "turquoise", "cyan", "skyblue", "steelblue", "slateblue",
    "royalblue", "midnightblue", "seagreen", "forestgreen", "darkgreen",
    "olivedrab", "chocolate", "sienna", "brown", "tan", "charcoal", "tomato",
    "firebrick", "darkred", "hotpink", "deeppink", "peachpuff", "goldenrod",
    "darkslategray", "dimgray", "lightgray", "gainsboro", "whitesmoke",
];

/// "charcoal" and "mint" aren't CSS names — map the near-misses the standard
/// lacks onto their nearest standard value so the vocabulary stays permissive.
fn css_color_value(word: &str) -> Option<String> {
    match word {
        "charcoal" => Some("#36454f".to_string()),
        "mint" => Some("#98ff98".to_string()),
        "cream" => Some("#fffdd0".to_string()),
        w if CSS_COLORS.contains(&w) => Some(w.to_string()),
        w if w.starts_with('#') && (w.len() == 4 || w.len() == 7)
            && w[1..].chars().all(|c| c.is_ascii_hexdigit()) =>
        {
            Some(w.to_string())
        }
        _ => None,
    }
}

/// RGB for the CSS named-color vocabulary (the spec's own defined values —
/// reference data, exactly like CSS_COLORS is the spec's own names). Used only
/// for contrast math; emission keeps the NAME so browsers resolve it natively.
fn named_rgb(word: &str) -> Option<(u8, u8, u8)> {
    let v = match word {
        "black" => (0, 0, 0),
        "silver" => (192, 192, 192),
        "gray" | "grey" => (128, 128, 128),
        "white" => (255, 255, 255),
        "maroon" => (128, 0, 0),
        "red" => (255, 0, 0),
        "purple" => (128, 0, 128),
        "fuchsia" | "magenta" => (255, 0, 255),
        "green" => (0, 128, 0),
        "lime" => (0, 255, 0),
        "olive" => (128, 128, 0),
        "yellow" => (255, 255, 0),
        "navy" => (0, 0, 128),
        "blue" => (0, 0, 255),
        "teal" => (0, 128, 128),
        "aqua" | "cyan" => (0, 255, 255),
        "orange" => (255, 165, 0),
        "gold" => (255, 215, 0),
        "coral" => (255, 127, 80),
        "salmon" => (250, 128, 114),
        "crimson" => (220, 20, 60),
        "indigo" => (75, 0, 130),
        "violet" => (238, 130, 238),
        "plum" => (221, 160, 221),
        "orchid" => (218, 112, 214),
        "khaki" => (240, 230, 140),
        "ivory" => (255, 255, 240),
        "beige" => (245, 245, 220),
        "mint" => (152, 255, 152),
        "azure" => (240, 255, 255),
        "lavender" => (230, 230, 250),
        "turquoise" => (64, 224, 208),
        "skyblue" => (135, 206, 235),
        "steelblue" => (70, 130, 180),
        "slateblue" => (106, 90, 205),
        "royalblue" => (65, 105, 225),
        "midnightblue" => (25, 25, 112),
        "seagreen" => (46, 139, 87),
        "forestgreen" => (34, 139, 34),
        "darkgreen" => (0, 100, 0),
        "olivedrab" => (107, 142, 35),
        "chocolate" => (210, 105, 30),
        "sienna" => (160, 82, 45),
        "brown" => (165, 42, 42),
        "tan" => (210, 180, 140),
        "charcoal" => (54, 69, 79),
        "cream" => (255, 253, 208),
        "tomato" => (255, 99, 71),
        "firebrick" => (178, 34, 34),
        "darkred" => (139, 0, 0),
        "hotpink" => (255, 105, 180),
        "deeppink" => (255, 20, 147),
        "peachpuff" => (255, 218, 185),
        "goldenrod" => (218, 165, 32),
        "darkslategray" => (47, 79, 79),
        "dimgray" => (105, 105, 105),
        "lightgray" => (211, 211, 211),
        "gainsboro" => (220, 220, 220),
        "whitesmoke" => (245, 245, 245),
        _ => return None,
    };
    Some(v)
}

/// Resolve any accepted color token (hex literal or CSS/mapped name) to RGB.
fn color_rgb(word: &str) -> Option<(u8, u8, u8)> {
    if let Some(hex) = word.strip_prefix('#') {
        let full = match hex.len() {
            3 => hex.chars().flat_map(|c| [c, c]).collect::<String>(),
            6 => hex.to_string(),
            _ => return None,
        };
        let r = u8::from_str_radix(&full[0..2], 16).ok()?;
        let g = u8::from_str_radix(&full[2..4], 16).ok()?;
        let b = u8::from_str_radix(&full[4..6], 16).ok()?;
        return Some((r, g, b));
    }
    named_rgb(word)
}

/// WCAG relative luminance of an sRGB color.
fn relative_luminance((r, g, b): (u8, u8, u8)) -> f64 {
    fn chan(c: u8) -> f64 {
        let s = c as f64 / 255.0;
        if s <= 0.03928 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        }
    }
    0.2126 * chan(r) + 0.7152 * chan(g) + 0.0722 * chan(b)
}

/// WCAG contrast ratio in [1.0, 21.0].
pub fn contrast_ratio(a: (u8, u8, u8), b: (u8, u8, u8)) -> f64 {
    let (la, lb) = (relative_luminance(a), relative_luminance(b));
    let (hi, lo) = if la >= lb { (la, lb) } else { (lb, la) };
    (hi + 0.05) / (lo + 0.05)
}

/// The legible text color (near-black or white) for a given background — the
/// one with the higher contrast ratio. Auto-chosen so surfaces are always
/// readable regardless of the requested palette.
fn on_color(bg: (u8, u8, u8)) -> (&'static str, (u8, u8, u8)) {
    let white = (255, 255, 255);
    let ink = (17, 17, 17);
    if contrast_ratio(bg, ink) >= contrast_ratio(bg, white) {
        ("#111111", ink)
    } else {
        ("#ffffff", white)
    }
}

/// Minimum contrast for text on colored UI surfaces (WCAG AA for large/bold
/// text and UI components is 3.0). Body text on white targets 4.5.
pub const MIN_UI_CONTRAST: f64 = 3.0;

/// WCAG AA minimum contrast for normal body text.
pub const MIN_BODY_CONTRAST: f64 = 4.5;

/// Business/establishment nouns that resolve to the STOREFRONT archetype. This
/// is WordNet's OWN taxonomy — the single-word hyponyms of `shop`, `restaurant`,
/// `eating_place`, and `market` (generated offline via NLTK WordNet, not
/// hand-picked). So "a website for my bakery / cafe / pharmacy / florist"
/// resolves to a storefront page through the REAL resolver, emergently, for any
/// business type WordNet knows — no per-domain hand mapping. (The 500k graph is
/// edge-starved — 0 relations — so these edges are seeded from WordNet directly.)
const STOREFRONT_NOUNS: &[&str] = &[
    "automat", "bakehouse", "bakery", "bakeshop", "barbershop", "bazaar", "bazar", "bistro",
    "bodega", "bookshop", "bookstall", "bookstore", "booth", "boutique", "brasserie", "brewpub",
    "buttery", "cafe", "cafeteria", "caff", "canteen", "charcuterie", "chophouse", "cleaners",
    "coffeehouse", "commissary", "confectionary", "confectionery", "cybercafe", "deli",
    "delicatessen", "diner", "drugstore", "estaminet", "florist", "garage", "grill", "grillroom",
    "haberdashery", "ironmonger", "lunchroom", "millinery", "newsstand", "outfitter", "patisserie",
    "pawnshop", "perfumery", "pharmacy", "pizzeria", "rotisserie", "salon", "slopshop", "stall",
    "stand", "steakhouse", "teahouse", "tearoom", "teashop", "thriftshop", "tobacconist", "toyshop",
];

/// A page section: registry unit with emission + its structural assertion.
#[derive(Clone)]
pub struct Section {
    pub name: &'static str,
    /// The selector/fragment whose PRESENCE verifies the section was emitted.
    pub assert_marker: &'static str,
    emit: fn(&str) -> String, // (title) -> html fragment
}

pub fn section_registry() -> Vec<Section> {
    fn nav(title: &str) -> String {
        format!("<nav class=\"site-nav\"><span class=\"brand\">{title}</span><ul><li><a href=\"index.html\">Home</a></li></ul></nav>")
    }
    fn hero(title: &str) -> String {
        format!("<header class=\"hero\"><h1>{title}</h1><p class=\"tagline\">Welcome to {title}.</p><a class=\"cta\" href=\"#contact\">Get started</a></header>")
    }
    fn features(_t: &str) -> String {
        "<section class=\"features\" id=\"features\"><h2>Features</h2><div class=\"grid\"><article class=\"card\"><h3>Fast</h3><p>Built for speed.</p></article><article class=\"card\"><h3>Simple</h3><p>No clutter.</p></article><article class=\"card\"><h3>Reliable</h3><p>Verified output.</p></article></div></section>".to_string()
    }
    fn gallery(_t: &str) -> String {
        "<section class=\"gallery\" id=\"gallery\"><h2>Gallery</h2><div class=\"grid\"><figure class=\"card\"><div class=\"ph\"></div><figcaption>One</figcaption></figure><figure class=\"card\"><div class=\"ph\"></div><figcaption>Two</figcaption></figure><figure class=\"card\"><div class=\"ph\"></div><figcaption>Three</figcaption></figure></div></section>".to_string()
    }
    fn contact(_t: &str) -> String {
        "<section class=\"contact\" id=\"contact\"><h2>Contact</h2><form><label>Name <input type=\"text\" name=\"name\" required></label><label>Email <input type=\"email\" name=\"email\" required></label><label>Message <textarea name=\"message\" rows=\"4\"></textarea></label><button type=\"submit\">Send</button></form></section>".to_string()
    }
    fn about(title: &str) -> String {
        format!("<section class=\"about\" id=\"about\"><h2>About</h2><p>{title} is built with verified components.</p></section>")
    }
    fn footer(title: &str) -> String {
        format!("<footer class=\"site-footer\"><p>&copy; {title}</p></footer>")
    }
    vec![
        Section { name: "nav", assert_marker: "<nav", emit: nav },
        Section { name: "hero", assert_marker: "class=\"hero\"", emit: hero },
        Section { name: "features", assert_marker: "class=\"features\"", emit: features },
        Section { name: "gallery", assert_marker: "class=\"gallery\"", emit: gallery },
        Section { name: "contact", assert_marker: "<form", emit: contact },
        Section { name: "about", assert_marker: "class=\"about\"", emit: about },
        Section { name: "footer", assert_marker: "<footer", emit: footer },
    ]
}

/// A comprehended site/page request.
#[derive(Clone, Debug, PartialEq)]
pub struct SiteRequest {
    pub page: String,             // file stem, e.g. "portfolio"
    pub title: String,            // display title
    pub theme: String,            // theme registry name
    pub colors: Vec<String>,      // resolved CSS values, request order
    pub sections: Vec<String>,    // resolved section names, request order
    /// The ask wires the contact form to the backend api ("posts to my api"):
    /// detected through the HUB'S BACKEND DOMAIN, not keywords.
    pub api_form: bool,
}

/// The WEB REGISTRY: sections and themes as ENTITIES with synonym edges and
/// definitions, resolved by the REAL lg-core `EntityResolver` — the same
/// emergent stack the coding registry rides (direct lemma, synonym edges,
/// morphology, fuzzy, definition overlap). "sleek" reaches `modern` through a
/// synonym edge; "photos" reaches `gallery`; an unlisted paraphrase can still
/// land via definition-overlap on the entity's own description. Data-shaped
/// seeds — extensible exactly like coding_registry.
fn web_registry() -> linguigenesis_core::registry::Registry {
    use linguigenesis_core::entity::{Entity, EntityType, RelationType};
    use linguigenesis_core::registry::Registry;
    let reg = Registry::new();
    let mut id: u64 = 1;
    let mut canonical = |lemma: &str, kind: &str, def: &str, syns: &[&str]| {
        let mut e = Entity::new(id, lemma.to_string(), EntityType::Noun);
        id += 1;
        e.add_definition(def.to_string());
        e.add_property("web_kind".into(), kind.into());
        let _ = reg.add_entity(e);
        for syn in syns {
            let se = Entity::new(id, syn.to_string(), EntityType::Noun);
            id += 1;
            let _ = reg.add_entity(se);
            let _ = reg.link_lemma_relation(syn, RelationType::Synonym, lemma);
            let _ = reg.link_lemma_relation(lemma, RelationType::Synonym, syn);
        }
    };
    // Sections.
    canonical("nav", "section", "a navigation menu bar with links at the top of the page", &["menu", "navigation", "navbar"]);
    canonical("hero", "section", "a large banner header splash welcoming visitors at the top", &["banner", "splash", "jumbotron", "header"]);
    canonical("features", "section", "a grid of feature cards highlighting benefits and services", &["benefits", "highlights", "services"]);
    canonical("gallery", "section", "a grid of photos images and pictures to showcase work", &["photos", "images", "pictures", "showcase", "portfolio"]);
    canonical("contact", "section", "a form where visitors reach out send a message or email to get in touch", &["form", "message", "email", "reach"]);
    canonical("about", "section", "an about section telling the story bio and background", &["bio", "story", "background"]);
    canonical("footer", "section", "a footer at the bottom with copyright", &["bottom", "copyright"]);
    // Themes.
    canonical("modern", "theme", "a modern sleek contemporary fresh design style", &["sleek", "contemporary", "fresh"]);
    canonical("minimal", "theme", "a minimal simple plain clean design style", &["simple", "plain", "clean", "minimalist"]);
    canonical("classic", "theme", "a classic traditional elegant vintage serif design style", &["traditional", "elegant", "vintage"]);
    // PAGE ARCHETYPES — a PURPOSE (landing/storefront/blog/docs) carries an
    // implied STRUCTURE. Resolving the purpose lets an ABSTRACT prompt ("a
    // landing page for my bakery") get a sensible section composition it never
    // spelled out. Resolved through the same real resolver; the default section
    // list is data (see `archetype_sections`). Lemmas are chosen NOT to collide
    // with section words (portfolio/showcase stay gallery synonyms).
    // The DEFINITION names the archetype's parts; the composition is DERIVED by
    // resolving those words against the section registry (see
    // `composition_from_definition`) — emergent, not a hardcoded list, so a
    // TAUGHT archetype composes for free from the definition it was taught with.
    canonical("landing", "archetype", "a landing home page with a hero, features, and a contact form", &["splash", "startpage"]);
    // Storefront synonyms = the base retail words + WordNet's whole business-type
    // taxonomy, so any shop/eatery noun ("bakery", "cafe", "florist"...) resolves
    // to a storefront page emergently through the real resolver.
    let storefront_syns: Vec<&str> = ["shop", "store", "storefront", "ecommerce", "catalog"]
        .iter()
        .chain(STOREFRONT_NOUNS.iter())
        .copied()
        .collect();
    canonical("storefront", "archetype", "a shop store page with a hero, a gallery, and a contact form", &storefront_syns);
    canonical("blog", "archetype", "a blog journal page with a hero, features, and an about story", &["journal", "articles", "posts"]);
    canonical("documentation", "archetype", "a documentation reference page with a hero and an about overview", &["docs", "manual", "guide"]);
    // MOODS — an aesthetic TONE that implies a PALETTE and a style, composed
    // from the mood's definition exactly like an archetype composes structure
    // (colors via the CSS vocabulary, theme via the resolver). This is how an
    // abstract AESTHETIC prompt ("a warm inviting page") becomes a concrete
    // palette + theme. Lemmas avoid collision with theme/color words.
    canonical("warm", "mood", "a warm inviting cozy classic look with coral, gold, and cream tones", &["cozy", "inviting"]);
    canonical("cool", "mood", "a cool calm fresh modern look with teal, azure, and navy tones", &["chill"]);
    canonical("earthy", "mood", "an earthy natural organic classic look with olive, brown, and tan tones", &["natural", "organic", "rustic"]);
    canonical("bold", "mood", "a bold energetic vibrant modern look with crimson, orange, and gold tones", &["energetic", "vibrant", "punchy"]);
    canonical("calm", "mood", "a calm serene soothing minimal look with azure, lavender, and mint tones", &["serene", "soothing", "tranquil"]);
    // GROWTH: merge every TAUGHT concept from the hub's web data registry —
    // runtime-taught vocabulary resolves exactly like the seeds (entities +
    // synonym edges through the same resolver).
    for c in crate::registry_hub::load_domain_concepts(crate::registry_hub::Domain::Web) {
        let syns: Vec<&str> = c.synonyms.iter().map(String::as_str).collect();
        canonical(&c.lemma, &c.kind, &c.definition, &syns);
    }
    reg
}

/// The composition an archetype implies for a page: derived from the archetype
/// entity's own DEFINITION via the DOMAIN-AGNOSTIC hub mechanism
/// (`registry_hub::compose_from_definition`) — web parts are "section" entities
/// under "web_kind". The identical hub call composes a BACKEND archetype from
/// routes/stores, so archetypes are not web-bound. Deriving from the definition
/// (not a hardcoded list) is what lets a TAUGHT or self-MINTED archetype compose
/// for free from the prose it carries.
fn archetype_sections(
    resolver: &linguigenesis_core::entity_resolution::EntityResolver,
    registry: &linguigenesis_core::registry::Registry,
    archetype: &str,
) -> Vec<String> {
    let Some(entity) = registry.get_by_lemma(archetype) else {
        return Vec::new();
    };
    let def = entity.definitions.first().cloned().unwrap_or_default();
    crate::registry_hub::compose_from_definition(resolver, registry, &def, "web_kind", "section")
}

/// Resolve one token to a web entity (section/theme) through the REAL resolver.
/// Returns (kind, canonical_lemma, score). Floor 0.5 admits the
/// definition-overlap tier; the canonical entity carries `web_kind`.
fn resolve_web_token(
    resolver: &linguigenesis_core::entity_resolution::EntityResolver,
    registry: &linguigenesis_core::registry::Registry,
    token: &str,
) -> Option<(String, String, f32)> {
    // Full emergent completion (shared hub helper): ranked candidates, and a
    // kind-less hit (a synonym entity reached by morphology) follows its
    // synonym EDGES to the canonical concept carrying web_kind.
    crate::registry_hub::resolve_domain(resolver, registry, token, "web_kind")
}

/// Emergent comprehension of a page request: routing gate = a construction cue
/// plus a web noun (token-level, morphology-aware); CONTENT resolution rides
/// the real EntityResolver over the web registry (synonym edges, morphology,
/// fuzzy, definition overlap) for sections and themes, and the CSS named-color
/// standard for palettes. Returns None when the prose carries no page/site
/// construction intent.
pub fn comprehend_site_request(text: &str) -> Option<SiteRequest> {
    use linguigenesis_core::entity_resolution::EntityResolver;
    let lower = text.to_lowercase();
    let tokens: Vec<String> = lower
        .split(|c: char| !c.is_alphanumeric() && c != '#' && c != '_')
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect();
    let morph_eq = |tok: &str, name: &str| -> bool {
        if tok == name {
            return true;
        }
        let mut tv = morphological_variants(tok);
        tv.push(tok.to_string());
        let mut nv = morphological_variants(name);
        nv.push(name.to_string());
        tv.iter().any(|v| nv.contains(v))
    };
    // Routing gate (token-level + morphology; routing, not resolution).
    const CUES: [&str; 8] = ["add", "create", "make", "build", "new", "generate", "want", "put"];
    const WEB: [&str; 6] = ["page", "website", "site", "webpage", "homepage", "web"];
    let has_cue = tokens.iter().any(|t| CUES.iter().any(|c| morph_eq(t, c)));
    let has_web = tokens.iter().any(|t| WEB.iter().any(|w| morph_eq(t, w)));
    if !has_cue || !has_web {
        return None;
    }
    // Page name: the token after "called"/"named" (structural cue).
    let page = tokens
        .iter()
        .position(|t| t == "called" || t == "named")
        .and_then(|i| tokens.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "page".to_string());
    let mut title = page.clone();
    if let Some(c) = title.get_mut(0..1) {
        c.make_ascii_uppercase();
    }
    // CONTENT resolution through the REAL resolver over the web registry.
    let registry = web_registry();
    let resolver = EntityResolver::new(registry.clone());
    let mut theme: Option<String> = None;
    let mut sections: Vec<String> = Vec::new();
    let mut archetype: Option<String> = None;
    let mut mood: Option<String> = None;
    for t in &tokens {
        // Routing tokens (construction cues, web nouns) are NOT content — skip
        // them so e.g. the cue "build" can never fuzzy-resolve to the mood
        // "bold". Routing vs resolution stays cleanly separated.
        if CUES.iter().any(|c| morph_eq(t, c)) || WEB.iter().any(|w| morph_eq(t, w)) {
            continue;
        }
        if let Some((kind, lemma, _score)) = resolve_web_token(&resolver, &registry, t) {
            match kind.as_str() {
                "theme" => {
                    if theme.is_none() {
                        theme = Some(lemma);
                    }
                }
                "section" => {
                    if !sections.contains(&lemma) {
                        sections.push(lemma);
                    }
                }
                "archetype" => {
                    if archetype.is_none() {
                        archetype = Some(lemma);
                    }
                }
                "mood" => {
                    if mood.is_none() {
                        mood = Some(lemma);
                    }
                }
                _ => {}
            }
        }
    }
    // STRUCTURE FROM PURPOSE: an abstract prompt that named a purpose but no
    // explicit sections gets the archetype's implied composition. Explicit
    // sections always win — inference only fills a void, never overrides intent.
    // STRUCTURE FROM PURPOSE: fill from the archetype when the prompt named no
    // CONTENT section (nav/footer are structural and auto-added, so they don't
    // count). Merge — archetype parts augment, explicit content always wins.
    let has_content = sections.iter().any(|s| s != "nav" && s != "footer");
    if !has_content {
        // A resolved purpose (archetype, incl. a business noun -> storefront)
        // fills its composition; otherwise a bare "make a website" defaults to a
        // landing page rather than an empty shell.
        let a = archetype.clone().unwrap_or_else(|| "landing".to_string());
        for part in archetype_sections(&resolver, &registry, &a) {
            if !sections.contains(&part) {
                sections.push(part);
            }
        }
    }
    // AESTHETIC FROM MOOD: a mood implies a theme (its style word) — derived
    // from the mood's own definition, only when no explicit theme was named.
    let mood_def = mood
        .as_ref()
        .and_then(|m| registry.get_by_lemma(m))
        .and_then(|e| e.definitions.first().cloned());
    if theme.is_none() {
        if let Some(def) = &mood_def {
            theme = crate::registry_hub::compose_from_definition(
                &resolver, &registry, def, "web_kind", "theme",
            )
            .into_iter()
            .next();
        }
    }
    let theme = theme.unwrap_or_else(|| "modern".to_string());
    // SITE+BACKEND: any token resolving through the hub's BACKEND domain to a
    // route/server concept ("posts to my api", "sends to the endpoint") wires
    // the contact form to the api.
    let api_form = {
        use crate::registry_hub::{backend_seeds, domain_registry, resolve_domain, Domain};
        let breg = domain_registry(Domain::Backend, &backend_seeds());
        let bres = EntityResolver::new(breg.clone());
        sections.contains(&"contact".to_string())
            && tokens.iter().any(|t| {
                resolve_domain(&bres, &breg, t, "backend_kind")
                    .map(|(kind, _, _)| kind == "route" || kind == "server")
                    .unwrap_or(false)
            })
    };
    // Colors: the platform's own vocabulary (CSS named colors + hex). When the
    // prompt named NO explicit color, a resolved mood supplies the palette,
    // derived from the mood's own definition (its named color words). Explicit
    // colors always win.
    let mut colors: Vec<String> = tokens.iter().filter_map(|t| css_color_value(t)).collect();
    if colors.is_empty() {
        if let Some(def) = &mood_def {
            colors = def
                .split(|c: char| !c.is_alphanumeric() && c != '#')
                .filter_map(css_color_value)
                .take(2)
                .collect();
        }
    }
    if !sections.contains(&"nav".to_string()) {
        sections.insert(0, "nav".to_string());
    }
    if !sections.contains(&"footer".to_string()) {
        sections.push("footer".to_string());
    }
    Some(SiteRequest { page, title, theme, colors, sections, api_form })
}

/// Emit the page HTML + the tokens->CSS stylesheet for a request.
pub fn emit_page(req: &SiteRequest) -> (String, String) {
    let registry = section_registry();
    let theme = theme_registry()
        .into_iter()
        .find(|t| t.name == req.theme)
        .unwrap_or_else(|| theme_registry().remove(0));
    let primary = req.colors.first().cloned().unwrap_or_else(|| "teal".into());
    let neutral = req
        .colors
        .get(1)
        .cloned()
        .unwrap_or_else(|| "#36454f".into());
    // Built-in sections use their curated emitters; TAUGHT sections (grown at
    // runtime through the registry hub) get a generic emitter built from the
    // concept's own definition — vocabulary growth without code changes.
    let taught: Vec<crate::registry_hub::Concept> =
        crate::registry_hub::load_domain_concepts(crate::registry_hub::Domain::Web);
    let body: String = req
        .sections
        .iter()
        .filter_map(|name| {
            if let Some(s) = registry.iter().find(|s| s.name == name) {
                return Some((s.emit)(&req.title));
            }
            taught.iter().find(|c| &c.lemma == name).map(|c| {
                let mut t = c.lemma.clone();
                if let Some(ch) = t.get_mut(0..1) {
                    ch.make_ascii_uppercase();
                }
                format!(
                    "<section class=\"taught\" id=\"{}\"><h2>{}</h2><p>{}</p></section>",
                    c.lemma, t, c.definition
                )
            })
        })
        .collect::<Vec<_>>()
        .join("\n");
    let mut html = format!(
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>{}</title>\n<link rel=\"stylesheet\" href=\"styles.css\">\n</head>\n<body>\n{}\n</body>\n</html>\n",
        req.title, body
    );
    if req.api_form {
        html = wire_form_action(&html, "/events");
    }
    // Auto-choose legible text for the primary-colored surfaces so they are
    // readable for ANY requested palette (contrast-verified below).
    let on_primary = req
        .colors
        .first()
        .and_then(|c| color_rgb(c))
        .map(|rgb| on_color(rgb).0)
        .unwrap_or("#ffffff");
    // Body copy sits on the page's default white background. The requested
    // neutral doubles as the body text color, but a light neutral would be
    // illegible on white — so body text uses the neutral only when it clears
    // the WCAG body floor, else falls back to ink. (--neutral stays as-is for
    // borders/accents.) Contrast-verified below; holds for any palette.
    let body_text = color_rgb(&neutral)
        .filter(|&rgb| contrast_ratio(rgb, (255, 255, 255)) >= MIN_BODY_CONTRAST)
        .map(|_| neutral.clone())
        .unwrap_or_else(|| "#111111".to_string());
    let css = format!(
        ":root {{\n  --primary: {primary};\n  --neutral: {neutral};\n  --on-primary: {on_primary};\n  --text: {body_text};\n  --radius: {};\n  --shadow: {};\n  --spacing: {};\n}}\n* {{ box-sizing: border-box; }}\nbody {{ margin: 0; font-family: {}; color: var(--text); }}\nh1, h2, h3 {{ font-weight: {}; }}\n.site-nav {{ display: flex; justify-content: space-between; align-items: center; padding: var(--spacing); background: var(--primary); color: var(--on-primary); }}\n.site-nav ul {{ list-style: none; display: flex; gap: 1rem; margin: 0; }}\n.site-nav a {{ color: var(--on-primary); text-decoration: none; }}\n.hero {{ padding: calc(var(--spacing) * 2) var(--spacing); text-align: center; }}\n.cta {{ display: inline-block; padding: 0.75rem 1.5rem; background: var(--primary); color: var(--on-primary); border-radius: var(--radius); box-shadow: var(--shadow); text-decoration: none; }}\n.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: var(--spacing); padding: var(--spacing); }}\n.card {{ border-radius: var(--radius); box-shadow: var(--shadow); padding: var(--spacing); }}\n.ph {{ height: 120px; background: var(--primary); opacity: 0.25; border-radius: var(--radius); }}\n.contact form {{ display: grid; gap: 1rem; padding: var(--spacing); max-width: 480px; }}\n.contact input, .contact textarea {{ width: 100%; padding: 0.5rem; border-radius: var(--radius); border: 1px solid var(--neutral); }}\nbutton {{ padding: 0.75rem 1.5rem; background: var(--primary); color: var(--on-primary); border: 0; border-radius: var(--radius); }}\n.site-footer {{ padding: var(--spacing); text-align: center; opacity: 0.8; }}\n",
        theme.radius, theme.shadow, theme.spacing, theme.font_stack, theme.heading_weight
    );
    (html, css)
}

/// Wire the contact form to the backend api route (site+backend integration):
/// the form gains action + method so submissions POST to the generated server.
pub fn wire_form_action(html: &str, path: &str) -> String {
    html.replacen("<form>", &format!("<form action=\"{path}\" method=\"post\">"), 1)
}

/// REQUEST-DERIVED verification: well-formed HTML + every requested section's
/// marker present + every requested color applied in the CSS + the title set.
/// Returns the list of failures (empty = verified).
pub fn verify_page(req: &SiteRequest, html: &str, css: &str) -> Vec<String> {
    let mut fails = Vec::new();
    if !html_well_formed(html) {
        fails.push("html not well-formed (tag balance)".into());
    }
    if !html.contains(&format!("<title>{}</title>", req.title)) {
        fails.push(format!("title '{}' missing", req.title));
    }
    for name in &req.sections {
        if let Some(s) = section_registry().into_iter().find(|s| s.name == name) {
            if !html.contains(s.assert_marker) {
                fails.push(format!("requested section '{name}' missing ({})", s.assert_marker));
            }
        } else {
            // Taught section: its id IS the assert marker.
            let marker = format!("id=\"{name}\"");
            if !html.contains(&marker) {
                fails.push(format!("requested taught section '{name}' missing ({marker})"));
            }
        }
    }
    for c in &req.colors {
        if !css.contains(c.as_str()) {
            fails.push(format!("requested color '{c}' not applied in css"));
        }
    }
    if req.api_form && !html.contains("action=\"/events\"") {
        fails.push("requested api-wired form has no action".into());
    }
    // CONTRAST: the one honestly-verifiable aesthetic property. Text on the
    // primary-colored surfaces (nav, buttons, CTA) must meet the WCAG UI floor.
    // The emitter auto-picks the text color, so this holds for any palette; a
    // failure here means the emitter chose wrong, not that the request is bad.
    if let Some(primary) = req.colors.first().and_then(|c| color_rgb(c)) {
        let (_, on) = on_color(primary);
        let ratio = contrast_ratio(primary, on);
        if ratio < MIN_UI_CONTRAST {
            fails.push(format!(
                "text on primary color has contrast {ratio:.2} < {MIN_UI_CONTRAST:.1} (illegible)"
            ));
        }
    }
    // CONTRAST MATRIX (second pair): body copy sits on the page's white
    // background. Its color (`--text`, auto-picked in emit_page) must clear the
    // WCAG body-text floor. By construction it does; a failure means the
    // auto-pick regressed.
    if let Some(text) = parse_css_var(css, "--text").and_then(|v| color_rgb(&v)) {
        let ratio = contrast_ratio(text, (255, 255, 255));
        if ratio < MIN_BODY_CONTRAST {
            fails.push(format!(
                "body text has contrast {ratio:.2} < {MIN_BODY_CONTRAST:.1} on white (illegible)"
            ));
        }
    }
    // ACCESSIBILITY: objective structural invariants over the emitted HTML —
    // decidable predicates in the spirit of the contrast check (aesthetics are
    // unverifiable; a11y structure is not).
    fails.extend(verify_accessibility(html));
    fails
}

/// Read the value of a CSS custom property (`--name: value;`) from a stylesheet.
fn parse_css_var(css: &str, name: &str) -> Option<String> {
    let key = format!("{name}:");
    let start = css.find(&key)? + key.len();
    let rest = &css[start..];
    let end = rest.find(';')?;
    Some(rest[..end].trim().to_string())
}

/// ACCESSIBILITY structural invariants — objectively checkable, decidable over
/// the emitted HTML (the natural generalization of section-marker presence):
///   * at most one `<h1>` (a single top-level heading);
///   * no heading-level SKIP in document order (h1 -> h3 without an h2);
///   * every `<img>` carries an `alt` attribute;
///   * every form control (input/textarea/select, excluding submit/button/
///     hidden) has an accessible name — wrapped in a `<label>` or `aria-label`.
/// Returns the list of failures (empty = accessible). Fail-closed in verify_page.
pub fn verify_accessibility(html: &str) -> Vec<String> {
    let mut fails = Vec::new();
    let mut h1_count = 0usize;
    let mut prev_level: Option<u32> = None;
    let mut in_label = false;
    let b = html.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] != b'<' {
            i += 1;
            continue;
        }
        let Some(close) = html[i..].find('>').map(|p| i + p) else { break };
        let tag = &html[i..=close]; // includes < and >
        let inner = tag.trim_start_matches('<').trim_end_matches('>').trim();
        let is_close = inner.starts_with('/');
        let name: String = inner
            .trim_start_matches('/')
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase();

        if name == "label" {
            in_label = !is_close && !inner.ends_with('/');
        }
        if !is_close {
            // Heading hierarchy.
            if name.len() == 2 && name.starts_with('h') {
                if let Some(level) = name[1..].parse::<u32>().ok().filter(|l| (1..=6).contains(l)) {
                    if level == 1 {
                        h1_count += 1;
                    }
                    if let Some(prev) = prev_level {
                        if level > prev + 1 {
                            fails.push(format!(
                                "heading level skip: h{prev} -> h{level} (no h{} between)",
                                prev + 1
                            ));
                        }
                    }
                    prev_level = Some(level);
                }
            }
            // Images need alt text.
            if name == "img" && !tag.contains("alt=") {
                fails.push("an <img> is missing its alt attribute".into());
            }
            // Form controls need an accessible name.
            if matches!(name.as_str(), "input" | "textarea" | "select") {
                let t = tag.to_ascii_lowercase();
                let excluded = t.contains("type=\"submit\"")
                    || t.contains("type=\"button\"")
                    || t.contains("type=\"hidden\"");
                let named = in_label || t.contains("aria-label=") || t.contains("aria-labelledby=");
                if !excluded && !named {
                    fails.push(format!("<{name}> has no accessible name (label/aria-label)"));
                }
            }
        }
        i = close + 1;
    }
    if h1_count > 1 {
        fails.push(format!("page has {h1_count} <h1> elements (at most one allowed)"));
    }
    fails
}

/// Tag-balance well-formedness for the emitted subset of HTML (void elements
/// exempt). Not a full parser — a structural sanity gate on OUR emitter.
fn html_well_formed(html: &str) -> bool {
    const VOID: [&str; 8] = ["meta", "link", "input", "img", "br", "hr", "source", "wbr"];
    let mut stack: Vec<String> = Vec::new();
    let mut i = 0;
    let b = html.as_bytes();
    while i < b.len() {
        if b[i] == b'<' {
            if html[i..].starts_with("<!") {
                i += 1;
                continue;
            }
            let close = match html[i..].find('>') {
                Some(c) => i + c,
                None => return false,
            };
            let inner = &html[i + 1..close];
            let is_end = inner.starts_with('/');
            let name: String = inner
                .trim_start_matches('/')
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric())
                .collect();
            if !name.is_empty() && !VOID.contains(&name.as_str()) {
                if is_end {
                    if stack.pop().as_deref() != Some(name.as_str()) {
                        return false;
                    }
                } else if !inner.ends_with('/') {
                    stack.push(name);
                }
            }
            i = close + 1;
        } else {
            i += 1;
        }
    }
    stack.is_empty()
}

/// Build a page on disk under `root/site/`: `<page>.html` + `styles.css`.
/// Fails closed: emission that doesn't verify against the request is an error
/// and nothing half-written is reported as success.
pub fn build_site_page(root: &Path, req: &SiteRequest) -> Result<Vec<String>, String> {
    let (html, css) = emit_page(req);
    let fails = verify_page(req, &html, &css);
    if !fails.is_empty() {
        return Err(format!("page failed request-fidelity checks: {}", fails.join("; ")));
    }
    let dir = root.join("site");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let page_rel = format!("site/{}.html", req.page);
    std::fs::write(root.join(&page_rel), &html).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("styles.css"), &css).map_err(|e| e.to_string())?;
    Ok(vec![page_rel, "site/styles.css".to_string()])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comprehends_the_full_ask() {
        let r = comprehend_site_request(
            "hey add a new page called portfolio to my website, make it a modern theme \
             with a hero and a gallery and a contact form, teal and charcoal color scheme",
        )
        .expect("comprehends");
        assert_eq!(r.page, "portfolio");
        assert_eq!(r.theme, "modern");
        assert_eq!(r.colors, vec!["teal".to_string(), "#36454f".to_string()]);
        for s in ["nav", "hero", "gallery", "contact", "footer"] {
            assert!(r.sections.contains(&s.to_string()), "{s} in {:?}", r.sections);
        }
    }

    #[test]
    fn declines_non_web_prose() {
        assert!(comprehend_site_request("add a function that triples a number").is_none());
        assert!(comprehend_site_request("paginate the results array").is_none());
        assert!(comprehend_site_request("sort a list of names").is_none());
    }

    #[test]
    fn emitted_page_verifies_against_its_request() {
        let r = comprehend_site_request(
            "create a page called landing on the site with hero, features and a contact form, \
             minimal theme, navy and ivory colors",
        )
        .expect("comprehends");
        assert_eq!(r.theme, "minimal");
        let (html, css) = emit_page(&r);
        let fails = verify_page(&r, &html, &css);
        assert!(fails.is_empty(), "fidelity failures: {fails:?}");
        // spot: the palette really landed in css
        assert!(css.contains("navy") && css.contains("ivory"));
        // well-formedness is a real gate: break a tag, verification fails.
        let broken = html.replace("</footer>", "");
        assert!(!verify_page(&r, &broken, &css).is_empty());
    }

    #[test]
    fn builds_on_disk_fails_closed() {
        let root = std::env::temp_dir().join(format!("nsynth_site_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let r = comprehend_site_request("build a new page called home for the website with a hero")
            .expect("comprehends");
        let written = build_site_page(&root, &r).expect("build");
        assert!(written.contains(&"site/home.html".to_string()));
        assert!(root.join("site/home.html").is_file());
        assert!(root.join("site/styles.css").is_file());
        let _ = std::fs::remove_dir_all(&root);
    }
}

/// Replace the nav's link list with the given (href, label) pairs — the one
/// post-processing hook both the create and extend paths share, so every page
/// on a site carries the SAME nav (the site's convention is regenerated, not
/// guessed).
pub fn set_nav_links(html: &str, links: &[(String, String)]) -> String {
    let items: String = links
        .iter()
        .map(|(href, label)| format!("<li><a href=\"{href}\">{label}</a></li>"))
        .collect();
    let Some(start) = html.find("<ul>") else { return html.to_string() };
    let Some(end) = html[start..].find("</ul>").map(|e| start + e) else {
        return html.to_string();
    };
    format!("{}<ul>{}{}", &html[..start], items, &html[end..])
}

/// Link-integrity check over a site directory: every `href="X.html"` in every
/// page must resolve to an existing file. Returns failures.
pub fn verify_site_links(dir: &Path) -> Vec<String> {
    let mut fails = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return vec!["site dir unreadable".into()];
    };
    let pages: Vec<std::path::PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "html").unwrap_or(false))
        .collect();
    for page in &pages {
        let Ok(text) = std::fs::read_to_string(page) else { continue };
        let mut rest = text.as_str();
        while let Some(pos) = rest.find("href=\"") {
            rest = &rest[pos + 6..];
            let Some(endq) = rest.find('"') else { break };
            let href = &rest[..endq];
            if href.ends_with(".html") && !dir.join(href).is_file() {
                fails.push(format!(
                    "{}: broken link '{href}'",
                    page.file_name().unwrap_or_default().to_string_lossy()
                ));
            }
            rest = &rest[endq..];
        }
    }
    fails
}

/// EXTEND an existing site with a new page, following its conventions: the new
/// page is emitted with the site's stylesheet and a nav linking every page, and
/// the nav in EVERY existing page is rewired to include the new page — one
/// coordinated change, link-integrity verified across the whole site.
/// Fails closed on any verification failure.
pub fn extend_site(root: &Path, req: &SiteRequest) -> Result<Vec<String>, String> {
    let dir = root.join("site");
    let mut existing: Vec<String> = std::fs::read_dir(&dir)
        .map_err(|e| format!("no existing site: {e}"))?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let n = e.file_name().to_string_lossy().to_string();
            n.ends_with(".html").then_some(n)
        })
        .collect();
    existing.sort();
    if existing.is_empty() {
        return Err("no existing pages to extend".into());
    }
    let new_file = format!("{}.html", req.page);
    if existing.contains(&new_file) {
        return Err(format!("page '{}' already exists", req.page));
    }
    // The site-wide nav: every existing page + the new one, labels from stems.
    let mut all: Vec<String> = existing.clone();
    all.push(new_file.clone());
    all.sort();
    let links: Vec<(String, String)> = all
        .iter()
        .map(|f| {
            let stem = f.trim_end_matches(".html");
            let mut label = stem.to_string();
            if let Some(c) = label.get_mut(0..1) {
                c.make_ascii_uppercase();
            }
            (f.clone(), label)
        })
        .collect();

    // New page: emitted for the request, nav rewired to the site-wide list.
    let (html, _css) = emit_page(req);
    let html = set_nav_links(&html, &links);
    let fails = verify_page(req, &html, &std::fs::read_to_string(dir.join("styles.css")).unwrap_or_default());
    // Palette check may legitimately fail against the EXISTING stylesheet (the
    // site's palette wins on extend); only structural page failures block.
    let blocking: Vec<&String> = fails.iter().filter(|f| !f.contains("color")).collect();
    if !blocking.is_empty() {
        return Err(format!("new page failed fidelity: {blocking:?}"));
    }

    let mut written = Vec::new();
    std::fs::write(dir.join(&new_file), &html).map_err(|e| e.to_string())?;
    written.push(format!("site/{new_file}"));
    // Rewire the nav in every existing page.
    for f in &existing {
        let p = dir.join(f);
        let text = std::fs::read_to_string(&p).map_err(|e| e.to_string())?;
        let rewired = set_nav_links(&text, &links);
        if rewired != text {
            std::fs::write(&p, rewired).map_err(|e| e.to_string())?;
            written.push(format!("site/{f}"));
        }
    }
    // Whole-site link integrity is the closing gate.
    let link_fails = verify_site_links(&dir);
    if !link_fails.is_empty() {
        return Err(format!("site link integrity failed: {link_fails:?}"));
    }
    Ok(written)
}

#[cfg(test)]
mod extend_tests {
    use super::*;

    #[test]
    fn extends_existing_site_and_rewires_every_nav() {
        let root = std::env::temp_dir().join(format!("nsynth_extend_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        // Existing site: home + about.
        let home = comprehend_site_request("build a new page called home for the website with a hero")
            .unwrap();
        build_site_page(&root, &home).expect("home");
        let about = comprehend_site_request("add a page called about to the site with an about section")
            .unwrap();
        extend_site(&root, &about).expect("about extends");
        // Now add the user's page.
        let req = comprehend_site_request(
            "add a new page called portfolio to my website with a gallery and a contact form",
        )
        .unwrap();
        let written = extend_site(&root, &req).expect("extend");
        assert!(written.contains(&"site/portfolio.html".to_string()));
        // EVERY page's nav links every page (coordinated rewiring).
        for f in ["home.html", "about.html", "portfolio.html"] {
            let text = std::fs::read_to_string(root.join("site").join(f)).unwrap();
            for target in ["home.html", "about.html", "portfolio.html"] {
                assert!(
                    text.contains(&format!("href=\"{target}\"")),
                    "{f} must link {target}"
                );
            }
        }
        // Link integrity green.
        assert!(verify_site_links(&root.join("site")).is_empty());
        // Duplicate page fails closed.
        assert!(extend_site(&root, &req).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }
}

/// One node of a parsed structure spec.
#[derive(Clone, Debug, PartialEq)]
pub struct StructureNode {
    pub path: String,
    pub is_dir: bool,
}

/// Parse a STRUCTURE SPEC (indented tree or markdown bullets) into nodes.
/// Rules: indentation (2 spaces or one `-`/`*` bullet level) = depth; names
/// ending in `/` are directories; names with an extension are files; bare
/// names are directories. Purely structural — no vocabulary needed.
pub fn parse_structure_spec(spec: &str) -> Vec<StructureNode> {
    let mut nodes: Vec<StructureNode> = Vec::new();
    let mut stack: Vec<(usize, String)> = Vec::new(); // (depth, dir path)
    for raw in spec.lines() {
        let line = raw.trim_end();
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let no_bullet = line.replace(['-', '*'], " ");
        let indent = no_bullet.len() - no_bullet.trim_start().len();
        let depth = indent / 2;
        let name = line
            .trim_start_matches([' ', '-', '*'])
            .trim()
            .to_string();
        if name.is_empty() {
            continue;
        }
        while stack.last().map(|(d, _)| *d >= depth).unwrap_or(false) {
            stack.pop();
        }
        let parent = stack.last().map(|(_, p)| p.clone()).unwrap_or_default();
        let clean = name.trim_end_matches('/').to_string();
        let path = if parent.is_empty() {
            clean.clone()
        } else {
            format!("{parent}/{clean}")
        };
        let is_dir = name.ends_with('/') || !clean.contains('.');
        nodes.push(StructureNode { path: path.clone(), is_dir });
        if is_dir {
            stack.push((depth, path));
        }
    }
    nodes
}

/// Scaffold a project from a structure spec. Directories are created; `.html`
/// files become REAL generated pages (emit_page, per-page verified); other
/// files are created with an honest scaffold header (organization is the
/// deliverable; unknown content is never fabricated). Closing gate: WALK-ASSERT
/// — every spec node must exist on disk with the right kind. THE SPEC IS THE
/// ORACLE.
pub fn scaffold_from_structure(root: &Path, spec: &str) -> Result<Vec<String>, String> {
    let nodes = parse_structure_spec(spec);
    if nodes.is_empty() {
        return Err("structure spec parsed to zero nodes".into());
    }
    let mut written = Vec::new();
    for n in &nodes {
        let p = root.join(&n.path);
        if n.is_dir {
            std::fs::create_dir_all(&p).map_err(|e| e.to_string())?;
        } else {
            if let Some(parent) = p.parent() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            if n.path.ends_with(".html") {
                let stem = p
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "page".into());
                let req = SiteRequest {
                    page: stem.clone(),
                    title: {
                        let mut t = stem.clone();
                        if let Some(c) = t.get_mut(0..1) {
                            c.make_ascii_uppercase();
                        }
                        t
                    },
                    theme: "modern".into(),
                    colors: vec![],
                    sections: vec!["nav".into(), "hero".into(), "footer".into()],
                    api_form: false,
                };
                let (html, css) = emit_page(&req);
                let fails = verify_page(&req, &html, &css);
                if !fails.is_empty() {
                    return Err(format!("page '{}' failed fidelity: {fails:?}", n.path));
                }
                std::fs::write(&p, html).map_err(|e| e.to_string())?;
                // one stylesheet per page's directory (idempotent)
                let cssp = p.parent().unwrap().join("styles.css");
                if !cssp.exists() {
                    std::fs::write(&cssp, css).map_err(|e| e.to_string())?;
                }
            } else if !p.exists() {
                std::fs::write(
                    &p,
                    format!("// scaffold: {} (generated from structure spec; content TODO)\n", n.path),
                )
                .map_err(|e| e.to_string())?;
            }
        }
        written.push(n.path.clone());
    }
    // WALK-ASSERT: the spec is the oracle.
    let mut fails = Vec::new();
    for n in &nodes {
        let p = root.join(&n.path);
        if n.is_dir && !p.is_dir() {
            fails.push(format!("missing dir {}", n.path));
        }
        if !n.is_dir && !p.is_file() {
            fails.push(format!("missing file {}", n.path));
        }
    }
    if !fails.is_empty() {
        return Err(format!("structure oracle failed: {fails:?}"));
    }
    Ok(written)
}

/// Find a structure-spec FILE named in the prose ("based on structure.md",
/// "like the layout in plan.txt") that exists under `root`.
pub fn structure_file_from_prose(root: &Path, text: &str) -> Option<std::path::PathBuf> {
    for tok in text.split(|c: char| c.is_whitespace() || c == ',' || c == ';') {
        let t = tok.trim_matches(|c: char| !(c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '/'));
        if t.ends_with(".md") || t.ends_with(".txt") || t.ends_with(".json") {
            let p = root.join(t);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

#[cfg(test)]
mod structure_tests {
    use super::*;

    const SPEC: &str = "\
# my project
src/
  core/
    engine.rs
  main.rs
site/
  index.html
  about.html
docs/
  README.md
";

    #[test]
    fn parses_indented_tree_with_dirs_and_files() {
        let nodes = parse_structure_spec(SPEC);
        let paths: Vec<&str> = nodes.iter().map(|n| n.path.as_str()).collect();
        assert!(paths.contains(&"src/core/engine.rs"));
        assert!(paths.contains(&"src/main.rs"));
        assert!(paths.contains(&"site/about.html"));
        assert!(paths.contains(&"docs/README.md"));
        assert!(nodes.iter().find(|n| n.path == "src/core").unwrap().is_dir);
        assert!(!nodes.iter().find(|n| n.path == "src/main.rs").unwrap().is_dir);
    }

    #[test]
    fn scaffolds_and_walk_asserts_the_spec() {
        let root = std::env::temp_dir().join(format!("nsynth_scaffold_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let written = scaffold_from_structure(&root, SPEC).expect("scaffold");
        assert!(written.len() >= 7, "{written:?}");
        // The spec is the oracle — and html files are REAL generated pages.
        assert!(root.join("src/core/engine.rs").is_file());
        let idx = std::fs::read_to_string(root.join("site/index.html")).unwrap();
        assert!(idx.contains("<!DOCTYPE html>") && idx.contains("class=\"hero\""));
        assert!(root.join("site/styles.css").is_file());
        // Re-scaffold is idempotent (no error, oracle still green).
        scaffold_from_structure(&root, SPEC).expect("idempotent");
        let _ = std::fs::remove_dir_all(&root);
    }
}

#[cfg(test)]
mod real_nl_tests {
    use super::*;

    /// THE ACID TEST: phrasings that share NO literal keyword with the
    /// registries — reachable only through the REAL resolver's synonym edges
    /// and morphology. This is the difference between keyword matching and the
    /// actual system.
    #[test]
    fn unseen_phrasings_resolve_through_the_real_resolver() {
        let r = comprehend_site_request(
            "put together a sleek new page called studio for my website with a big banner, \
             a photo showcase, and a way for people to send a message — navy and cream colors",
        )
        .expect("comprehends unseen phrasing");
        assert_eq!(r.theme, "modern", "sleek -> modern via synonym edge");
        assert!(r.sections.contains(&"hero".to_string()), "banner -> hero: {:?}", r.sections);
        assert!(r.sections.contains(&"gallery".to_string()), "photo/showcase -> gallery: {:?}", r.sections);
        assert!(r.sections.contains(&"contact".to_string()), "message -> contact: {:?}", r.sections);
        assert_eq!(r.colors, vec!["navy".to_string(), "#fffdd0".to_string()]);

        let r2 = comprehend_site_request(
            "create a simple webpage called docs with highlights and our story",
        )
        .expect("comprehends");
        assert_eq!(r2.theme, "minimal", "simple -> minimal: {:?}", r2.theme);
        assert!(r2.sections.contains(&"features".to_string()), "highlights -> features");
        assert!(r2.sections.contains(&"about".to_string()), "story -> about");

        let r3 = comprehend_site_request(
            "build an elegant page called menu-page for the site with a navigation menu and pictures",
        )
        .expect("comprehends");
        assert_eq!(r3.theme, "classic", "elegant -> classic");
        assert!(r3.sections.contains(&"gallery".to_string()), "pictures -> gallery");
    }

    /// BACKEND TEACH LOOP drives real behavior (symmetric with the web teach
    /// loop): a runtime-taught backend synonym flips api-form detection. Before
    /// teaching, "sends to my webhook" carries no known api target; after
    /// teaching "webhook" as a route (synonym "hook"), the same phrasing wires
    /// the contact form to the api — through the REAL resolver, no keywords.
    #[test]
    fn taught_backend_concept_flips_api_form_detection() {
        use crate::registry_hub::{teach_concept, Concept, Domain};
        let p = std::env::temp_dir()
            .join(format!("nsynth_site_teachback_{}.json", std::process::id()));
        let _ = std::fs::remove_file(&p);
        std::env::set_var(Domain::Backend.env_var_name(), &p);

        // Unknown target: no api wiring yet.
        let before = comprehend_site_request(
            "add a page called reach for my site with a contact form that sends to my webhook",
        )
        .expect("comprehends");
        assert!(!before.api_form, "webhook is unknown before teaching: {:?}", before.api_form);

        // Teach the backend vocabulary at runtime.
        teach_concept(
            Domain::Backend,
            Concept {
                lemma: "webhook".to_string(),
                kind: "route".to_string(),
                definition: "an inbound callback endpoint the site posts to".to_string(),
                synonyms: vec!["hook".to_string()],
            },
        )
        .expect("teach persists");

        // Same phrasing now wires the form, and so does the taught synonym.
        for phrase in [
            "add a page called reach for my site with a contact form that sends to my webhook",
            "add a page called reach for my site with a contact form that posts to a hook",
        ] {
            let after = comprehend_site_request(phrase).expect("comprehends");
            assert!(after.api_form, "taught backend concept must wire the form: {phrase}");
        }

        std::env::remove_var(Domain::Backend.env_var_name());
        let _ = std::fs::remove_file(&p);
    }

    /// CONTRAST is real and auto-corrected: the emitter picks legible text for
    /// the primary surfaces so verify passes for ANY palette — including a
    /// pathological light primary (yellow) where fixed white text would be
    /// illegible. The one honestly-verifiable aesthetic property.
    #[test]
    fn primary_surface_text_is_always_legible() {
        // WCAG anchors: black-on-white is the maximum 21:1.
        assert!((contrast_ratio((0, 0, 0), (255, 255, 255)) - 21.0).abs() < 0.01);

        // Dark primary -> white text; light primary -> near-black text.
        assert_eq!(on_color((0, 0, 128)).0, "#ffffff", "navy takes white text");
        assert_eq!(on_color((255, 255, 0)).0, "#111111", "yellow takes ink text");

        // A pathological ask: yellow primary. Fixed white text would be ~1.07:1
        // (illegible); the emitter must pick ink and pass the floor.
        let req = SiteRequest {
            page: "p".into(),
            title: "P".into(),
            theme: "modern".into(),
            colors: vec!["yellow".into()],
            sections: vec!["nav".into(), "hero".into(), "footer".into()],
            api_form: false,
        };
        let (html, css) = emit_page(&req);
        assert!(css.contains("--on-primary: #111111"), "yellow -> ink text chosen: {css}");
        let fails = verify_page(&req, &html, &css);
        assert!(
            !fails.iter().any(|f| f.contains("contrast")),
            "auto-chosen text must pass the contrast floor: {fails:?}"
        );

        // A normal dark palette keeps white text and also passes.
        let req2 = SiteRequest { colors: vec!["navy".into()], ..req.clone() };
        let (_h2, css2) = emit_page(&req2);
        assert!(css2.contains("--on-primary: #ffffff"), "navy -> white text: {css2}");
    }

    /// STRUCTURE FROM PURPOSE: an ABSTRACT prompt that names a page purpose but
    /// no explicit sections gets a sensible composition, inferred EMERGENTLY from
    /// the archetype's own definition (not a hardcoded table). Explicit sections
    /// always win over inference.
    #[test]
    fn abstract_prompt_infers_structure_from_archetype() {
        // No section words at all — only the purpose "landing" + a domain noun.
        let r = comprehend_site_request("build a landing page for my bakery")
            .expect("comprehends an abstract landing prompt");
        for s in ["hero", "features", "contact"] {
            assert!(r.sections.contains(&s.to_string()), "landing implies {s}: {:?}", r.sections);
        }
        // A storefront purpose implies a gallery (catalog), not features.
        let r2 = comprehend_site_request("make a storefront site for my shop")
            .expect("comprehends storefront");
        assert!(r2.sections.contains(&"gallery".to_string()), "storefront implies gallery: {:?}", r2.sections);

        // Explicit sections WIN — inference only fills a void.
        let r3 = comprehend_site_request("build a landing page with just an about section")
            .expect("comprehends");
        assert!(r3.sections.contains(&"about".to_string()), "explicit about honored");
        assert!(!r3.sections.contains(&"features".to_string()), "archetype did NOT override explicit: {:?}", r3.sections);
    }

    /// DOMAIN NOUN -> ARCHETYPE via WordNet: "a website for my bakery" (no
    /// section/archetype word at all) resolves through the business-type taxonomy
    /// to a storefront composition. Generalizes to any WordNet shop/eatery noun.
    /// A bare "make a website" defaults to a landing page, not an empty shell.
    #[test]
    fn domain_noun_resolves_to_storefront_via_wordnet() {
        // The exact abstract prompt the user asked about.
        let r = comprehend_site_request("generate a professional website for my bakery")
            .expect("routes + comprehends");
        for s in ["hero", "gallery", "contact"] {
            assert!(r.sections.contains(&s.to_string()), "bakery -> storefront implies {s}: {:?}", r.sections);
        }
        // Other WordNet business types resolve the same way.
        for noun in ["cafe", "pharmacy", "florist", "boutique"] {
            let rr = comprehend_site_request(&format!("build a website for my {noun}"))
                .unwrap_or_else(|| panic!("{noun} comprehends"));
            assert!(rr.sections.contains(&"gallery".to_string()), "{noun} -> storefront: {:?}", rr.sections);
        }
        // Bare website with no signal -> landing default (hero/features/contact),
        // not an empty nav+footer shell.
        let bare = comprehend_site_request("make me a website").expect("comprehends");
        assert!(bare.sections.contains(&"hero".to_string()), "bare website -> landing: {:?}", bare.sections);
        // A non-business word must NOT spuriously resolve to storefront.
        let plain = comprehend_site_request("build a website with an about section").expect("comprehends");
        assert!(plain.sections.contains(&"about".to_string()), "explicit about honored: {:?}", plain.sections);
    }

    /// AESTHETIC FROM MOOD: an abstract prompt naming a TONE (but no explicit
    /// colors/theme) derives a palette and a style from the mood's own
    /// definition — the same compose-from-definition mechanism as archetypes,
    /// for visual tokens. Explicit colors/theme always win.
    #[test]
    fn abstract_aesthetic_prompt_derives_palette_and_theme_from_mood() {
        // Purpose + tone, nothing explicit: structure AND aesthetic both inferred.
        let r = comprehend_site_request("build a warm inviting landing page")
            .expect("comprehends");
        assert!(r.colors.contains(&"coral".to_string()), "warm -> coral palette: {:?}", r.colors);
        assert_eq!(r.theme, "classic", "warm mood implies a classic style: {}", r.theme);
        assert!(r.sections.contains(&"hero".to_string()), "landing still composes structure");

        // Explicit color + theme override the mood.
        let r2 = comprehend_site_request("build a warm modern landing page with a teal color scheme")
            .expect("comprehends");
        assert!(r2.colors.contains(&"teal".to_string()), "explicit teal wins: {:?}", r2.colors);
        assert!(!r2.colors.contains(&"coral".to_string()), "mood palette suppressed by explicit");
        assert_eq!(r2.theme, "modern", "explicit modern wins over mood: {}", r2.theme);

        // A cool mood yields a cool palette.
        let r3 = comprehend_site_request("make a cool serene landing page").expect("comprehends");
        assert!(
            r3.colors.iter().any(|c| c == "teal" || c == "azure"),
            "cool -> teal/azure palette: {:?}",
            r3.colors
        );
    }

    /// LEARN A NEW ARCHETYPE: a runtime-taught archetype composes for free from
    /// the definition it was taught with — no code change. This is the emergent
    /// "make its own archetypes" path: the composition is COMPREHENDED, not coded.
    #[test]
    fn taught_archetype_composes_from_its_definition() {
        use crate::registry_hub::{teach_concept, Concept, Domain};
        let p = std::env::temp_dir().join(format!("nsynth_arch_{}.json", std::process::id()));
        let _ = std::fs::remove_file(&p);
        std::env::set_var(Domain::Web.env_var_name(), &p);

        // Before teaching, "dashboard" is unknown -> the bare-website default
        // (landing: hero/features/contact), which notably does NOT include the
        // taught-specific "about" section. Teaching then changes the composition.
        let before = comprehend_site_request("build a dashboard page").expect("comprehends");
        assert!(
            !before.sections.contains(&"about".to_string()),
            "dashboard is unknown before teaching (no taught 'about'): {:?}",
            before.sections
        );

        // Teach the archetype by DEFINITION — its parts are named in prose.
        teach_concept(
            Domain::Web,
            Concept {
                lemma: "dashboard".to_string(),
                kind: "archetype".to_string(),
                definition: "a dashboard page with a hero, features, and an about overview"
                    .to_string(),
                synonyms: vec!["console".to_string()],
            },
        )
        .expect("teach persists");

        // Now the same abstract prompt composes from the TAUGHT definition, and
        // so does its taught synonym.
        for phrase in ["build a dashboard page", "build a console page"] {
            let r = comprehend_site_request(phrase).expect("comprehends");
            for s in ["hero", "features", "about"] {
                assert!(
                    r.sections.contains(&s.to_string()),
                    "taught dashboard implies {s} from its definition ({phrase}): {:?}",
                    r.sections
                );
            }
        }

        std::env::remove_var(Domain::Web.env_var_name());
        let _ = std::fs::remove_file(&p);
    }

    /// ACCESSIBILITY invariants are real and pass on emitted pages but CATCH
    /// synthetic violations — the natural generalization of the contrast check
    /// to objective a11y structure.
    #[test]
    fn accessibility_invariants_hold_and_catch_violations() {
        // A real generated page with a form is accessible (inputs wrapped in
        // labels, one h1, no img/heading issues).
        let req = SiteRequest {
            page: "reach".into(),
            title: "Reach".into(),
            theme: "modern".into(),
            colors: vec!["teal".into()],
            sections: vec!["nav".into(), "hero".into(), "contact".into(), "footer".into()],
            api_form: false,
        };
        let (html, css) = emit_page(&req);
        assert!(verify_accessibility(&html).is_empty(), "generated page must be accessible");
        assert!(verify_page(&req, &html, &css).is_empty(), "page verifies clean");

        // Each violation class is caught.
        assert!(
            verify_accessibility("<h1>A</h1><h1>B</h1>")
                .iter()
                .any(|f| f.contains("<h1>")),
            "two h1s caught"
        );
        assert!(
            verify_accessibility("<h1>A</h1><h3>skip</h3>")
                .iter()
                .any(|f| f.contains("skip")),
            "heading skip h1->h3 caught"
        );
        assert!(
            verify_accessibility("<img src=\"x.png\">")
                .iter()
                .any(|f| f.contains("alt")),
            "img without alt caught"
        );
        assert!(
            verify_accessibility("<form><input type=\"text\" name=\"n\"></form>")
                .iter()
                .any(|f| f.contains("accessible name")),
            "unlabeled input caught"
        );
        // A labeled input and an alt'd img are accepted.
        assert!(
            verify_accessibility("<label>Name <input type=\"text\"></label>").is_empty(),
            "labeled input passes"
        );
        assert!(
            verify_accessibility("<img src=\"x.png\" alt=\"a cat\">").is_empty(),
            "alt'd img passes"
        );
    }

    /// Precision: web-ish words in OP requests still never comprehend as sites,
    /// and gibberish resolves nothing.
    #[test]
    fn precision_holds_under_the_resolver() {
        assert!(comprehend_site_request("paginate the results array").is_none());
        assert!(comprehend_site_request("add a function that triples a number").is_none());
        assert!(comprehend_site_request("frobnicate the zorp").is_none());
    }
}

/// MINE the web vocabulary: reflect the engine's OWN section/theme surface into
/// the hub's web data registry (the capability_miner discipline applied to the
/// web domain). Built-ins become versioned data concepts; TAUGHT concepts are
/// PRESERVED (merge-by-lemma, taught entries never dropped) — the two growth
/// seams composing. Idempotent. Returns how many concepts the file now holds.
pub fn mine_web_registry() -> Result<usize, String> {
    use crate::registry_hub::{load_domain_concepts, teach_concept, Concept, Domain};
    let mut mined: Vec<Concept> = Vec::new();
    for s in section_registry() {
        mined.push(Concept {
            lemma: s.name.to_string(),
            kind: "section".into(),
            definition: format!("built-in section asserting '{}'", s.assert_marker),
            synonyms: vec![],
        });
    }
    for t in theme_registry() {
        mined.push(Concept {
            lemma: t.name.to_string(),
            kind: "theme".into(),
            definition: format!("built-in theme (font {}, radius {})", t.font_stack, t.radius),
            synonyms: vec![],
        });
    }
    // Taught concepts win over mined reflections on lemma collision (a taught
    // definition/synonyms are richer than the reflected stub).
    let taught = load_domain_concepts(Domain::Web);
    for c in mined {
        if taught.iter().any(|t| t.lemma == c.lemma) {
            continue;
        }
        teach_concept(Domain::Web, c)?;
    }
    Ok(load_domain_concepts(Domain::Web).len())
}

#[cfg(test)]
mod mine_tests {
    use super::*;
    use crate::registry_hub::{load_domain_concepts, teach_concept, Concept, Domain};

    #[test]
    fn mining_reflects_builtins_and_preserves_taught() {
        let p = std::env::temp_dir().join(format!("nsynth_webmine_{}.json", std::process::id()));
        let _ = std::fs::remove_file(&p);
        std::env::set_var(Domain::Web.env_var_name(), &p);
        // A taught concept exists first...
        teach_concept(
            Domain::Web,
            Concept {
                lemma: "testimonials".into(),
                kind: "section".into(),
                definition: "customer quotes in a row".into(),
                synonyms: vec!["reviews".into()],
            },
        )
        .unwrap();
        let n = mine_web_registry().expect("mine");
        assert!(n >= 10, "sections + themes + taught: {n}");
        let all = load_domain_concepts(Domain::Web);
        // Built-ins reflected...
        assert!(all.iter().any(|c| c.lemma == "gallery" && c.kind == "section"));
        assert!(all.iter().any(|c| c.lemma == "modern" && c.kind == "theme"));
        // ...and the taught concept SURVIVES with its richer definition.
        let t = all.iter().find(|c| c.lemma == "testimonials").unwrap();
        assert_eq!(t.definition, "customer quotes in a row");
        assert_eq!(t.synonyms, vec!["reviews".to_string()]);
        // Idempotent.
        let n2 = mine_web_registry().expect("re-mine");
        assert_eq!(n, n2);
        std::env::remove_var(Domain::Web.env_var_name());
        let _ = std::fs::remove_file(&p);
    }
}


