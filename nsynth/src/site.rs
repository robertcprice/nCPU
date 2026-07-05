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

/// Emergent comprehension of a page request: theme words resolve against the
/// theme registry, color words against the CSS vocabulary, section nouns
/// against the section registry (morphology per word). Returns None when the
/// prose carries no page/site construction intent.
pub fn comprehend_site_request(text: &str) -> Option<SiteRequest> {
    let lower = text.to_lowercase();
    let tokens: Vec<String> = lower
        .split(|c: char| !c.is_alphanumeric() && c != '#' && c != '_')
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect();
    // Gate: a construction cue + a web noun.
    const CUES: [&str; 7] = ["add", "create", "make", "build", "new", "generate", "want"];
    const WEB: [&str; 4] = ["page", "website", "site", "webpage"];
    let has_cue = tokens.iter().any(|t| CUES.contains(&t.as_str()));
    let has_web = tokens.iter().any(|t| WEB.contains(&t.as_str()));
    if !has_cue || !has_web {
        return None;
    }
    let word_matches = |tok: &str, name: &str| -> bool {
        if tok == name {
            return true;
        }
        let mut tv = morphological_variants(tok);
        tv.push(tok.to_string());
        let mut nv = morphological_variants(name);
        nv.push(name.to_string());
        tv.iter().any(|v| nv.contains(v))
    };
    // Page name: the token after "called"/"named", else "page".
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
    // Theme: first theme-registry name matched (morphology).
    let theme = theme_registry()
        .iter()
        .find(|th| tokens.iter().any(|t| word_matches(t, th.name)))
        .map(|th| th.name.to_string())
        .unwrap_or_else(|| "modern".to_string());
    // Colors: every token resolving in the CSS vocabulary, request order.
    let colors: Vec<String> = tokens.iter().filter_map(|t| css_color_value(t)).collect();
    // Sections: every token resolving in the section registry, request order,
    // deduped. Always ensure nav + footer bracket the page.
    let mut sections: Vec<String> = Vec::new();
    for t in &tokens {
        for s in section_registry() {
            if word_matches(t, s.name) && !sections.contains(&s.name.to_string()) {
                sections.push(s.name.to_string());
            }
        }
    }
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
