//! SITE DOMAIN through the REAL front door (handle_query): the user's literal
//! ask builds a verified page on disk; op requests are never hijacked.

use mog_synth::agent::{CodingAgentSession, GuardrailPolicy};
use std::fs;
use std::path::PathBuf;

fn fresh_root(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("nsynth_sitesess_{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("root");
    root
}

/// THE LITERAL ASK, end to end.
#[test]
fn handle_query_builds_page_from_the_literal_ask() {
    let root = fresh_root("literal");
    let mut s = CodingAgentSession::new(&root, GuardrailPolicy::default());
    let r = s.handle_query(
        "hey add a new page called portfolio to my website and make it modern theme \
         with a hero and a gallery and a contact form and a teal and charcoal color scheme",
    );
    assert!(r.success, "response: {}", r.response);
    assert_eq!(r.workflow, "site.build");
    let html = fs::read_to_string(root.join("site/portfolio.html")).expect("page on disk");
    let css = fs::read_to_string(root.join("site/styles.css")).expect("css on disk");
    // Request-fidelity spot checks (the builder already verified; re-prove here).
    assert!(html.contains("<title>Portfolio</title>"), "{html}");
    assert!(html.contains("class=\"hero\"") && html.contains("class=\"gallery\"") && html.contains("<form"));
    assert!(css.contains("teal") && css.contains("#36454f"), "palette applied: {css}");
    let _ = fs::remove_dir_all(&root);
}

/// HIJACK GUARDS: op requests with web-ish words route as before.
#[test]
fn handle_query_does_not_hijack_op_requests() {
    let root = fresh_root("nohijack");
    let mut s = CodingAgentSession::new(&root, GuardrailPolicy::default());
    for q in ["paginate the results array", "add a function that triples a number"] {
        let r = s.handle_query(q);
        assert_ne!(r.workflow, "site.build", "{q:?} must not build a site: {}", r.response);
        assert!(!root.join("site").exists(), "{q:?} wrote site files");
    }
    let _ = fs::remove_dir_all(&root);
}

/// EXTEND through the front door: a second ask against the SAME session root
/// follows the existing site's conventions — new page + every nav rewired.
#[test]
fn handle_query_extends_existing_site() {
    let root = fresh_root("extend");
    let mut s = CodingAgentSession::new(&root, GuardrailPolicy::default());
    let r1 = s.handle_query("make a new page called home for my website with a hero");
    assert!(r1.success, "{}", r1.response);
    let r2 = s.handle_query("add a new page called pricing to my website with features and a contact form");
    assert!(r2.success, "{}", r2.response);
    for f in ["home.html", "pricing.html"] {
        let text = fs::read_to_string(root.join("site").join(f)).expect(f);
        assert!(text.contains("href=\"home.html\"") && text.contains("href=\"pricing.html\""),
            "{f} nav rewired: {text}");
    }
    assert!(mog_synth::site::verify_site_links(&root.join("site")).is_empty());
    let _ = fs::remove_dir_all(&root);
}
