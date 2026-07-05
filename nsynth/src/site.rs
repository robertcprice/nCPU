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
    reg
}

/// Resolve one token to a web entity (section/theme) through the REAL resolver.
/// Returns (kind, canonical_lemma, score). Floor 0.5 admits the
/// definition-overlap tier; the canonical entity carries `web_kind`.
fn resolve_web_token(
    resolver: &linguigenesis_core::entity_resolution::EntityResolver,
    token: &str,
) -> Option<(String, String, f32)> {
    // Scan the RANKED candidates for the first that carries web_kind: a synonym
    // surface ("banner") direct-matches its own entity first, but the canonical
    // ("hero") arrives via the synonym-edge lens right behind it.
    resolver
        .rank_candidates(token)
        .into_iter()
        .filter(|r| r.evidence.score >= 0.5)
        .find_map(|r| {
            let kind = r.entity.get_property("web_kind")?.clone();
            Some((kind, r.entity.lemma.clone(), r.evidence.score))
        })
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
    let resolver = EntityResolver::new(web_registry());
    let mut theme: Option<String> = None;
    let mut sections: Vec<String> = Vec::new();
    for t in &tokens {
        if let Some((kind, lemma, _score)) = resolve_web_token(&resolver, t) {
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
                _ => {}
            }
        }
    }
    let theme = theme.unwrap_or_else(|| "modern".to_string());
    // Colors: the platform's own vocabulary (CSS named colors + hex).
    let colors: Vec<String> = tokens.iter().filter_map(|t| css_color_value(t)).collect();
    if !sections.contains(&"nav".to_string()) {
        sections.insert(0, "nav".to_string());
    }
    if !sections.contains(&"footer".to_string()) {
        sections.push("footer".to_string());
    }
    Some(SiteRequest { page, title, theme, colors, sections })
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
    let body: String = req
        .sections
        .iter()
        .filter_map(|name| registry.iter().find(|s| s.name == name))
        .map(|s| (s.emit)(&req.title))
        .collect::<Vec<_>>()
        .join("\n");
    let html = format!(
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n<title>{}</title>\n<link rel=\"stylesheet\" href=\"styles.css\">\n</head>\n<body>\n{}\n</body>\n</html>\n",
        req.title, body
    );
    let css = format!(
        ":root {{\n  --primary: {primary};\n  --neutral: {neutral};\n  --radius: {};\n  --shadow: {};\n  --spacing: {};\n}}\n* {{ box-sizing: border-box; }}\nbody {{ margin: 0; font-family: {}; color: var(--neutral); }}\nh1, h2, h3 {{ font-weight: {}; }}\n.site-nav {{ display: flex; justify-content: space-between; align-items: center; padding: var(--spacing); background: var(--primary); color: white; }}\n.site-nav ul {{ list-style: none; display: flex; gap: 1rem; margin: 0; }}\n.site-nav a {{ color: white; text-decoration: none; }}\n.hero {{ padding: calc(var(--spacing) * 2) var(--spacing); text-align: center; }}\n.cta {{ display: inline-block; padding: 0.75rem 1.5rem; background: var(--primary); color: white; border-radius: var(--radius); box-shadow: var(--shadow); text-decoration: none; }}\n.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: var(--spacing); padding: var(--spacing); }}\n.card {{ border-radius: var(--radius); box-shadow: var(--shadow); padding: var(--spacing); }}\n.ph {{ height: 120px; background: var(--primary); opacity: 0.25; border-radius: var(--radius); }}\n.contact form {{ display: grid; gap: 1rem; padding: var(--spacing); max-width: 480px; }}\n.contact input, .contact textarea {{ width: 100%; padding: 0.5rem; border-radius: var(--radius); border: 1px solid var(--neutral); }}\nbutton {{ padding: 0.75rem 1.5rem; background: var(--primary); color: white; border: 0; border-radius: var(--radius); }}\n.site-footer {{ padding: var(--spacing); text-align: center; opacity: 0.8; }}\n",
        theme.radius, theme.shadow, theme.spacing, theme.font_stack, theme.heading_weight
    );
    (html, css)
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
        }
    }
    for c in &req.colors {
        if !css.contains(c.as_str()) {
            fails.push(format!("requested color '{c}' not applied in css"));
        }
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

    /// Precision: web-ish words in OP requests still never comprehend as sites,
    /// and gibberish resolves nothing.
    #[test]
    fn precision_holds_under_the_resolver() {
        assert!(comprehend_site_request("paginate the results array").is_none());
        assert!(comprehend_site_request("add a function that triples a number").is_none());
        assert!(comprehend_site_request("frobnicate the zorp").is_none());
    }
}
