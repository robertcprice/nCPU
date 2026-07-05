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

/// THE SECOND LITERAL ASK: "make this new project and organize based on this
/// file containing structure" — the spec file is the oracle.
#[test]
fn handle_query_scaffolds_project_from_structure_file() {
    let root = fresh_root("scaffold");
    fs::write(
        root.join("structure.md"),
        "src/\n  core/\n    engine.rs\n  main.rs\nsite/\n  index.html\ndocs/\n  README.md\n",
    )
    .expect("spec");
    let mut s = CodingAgentSession::new(&root, GuardrailPolicy::default());
    let r = s.handle_query(
        "hey make this new project and organize it based on structure.md please",
    );
    assert!(r.success, "response: {}", r.response);
    assert_eq!(r.workflow, "site.scaffold");
    assert!(root.join("src/core/engine.rs").is_file());
    assert!(root.join("src/main.rs").is_file());
    assert!(root.join("docs/README.md").is_file());
    let idx = fs::read_to_string(root.join("site/index.html")).expect("real page");
    assert!(idx.contains("<!DOCTYPE html>"), "generated page, not a stub");
    let _ = fs::remove_dir_all(&root);
}

/// No spec file -> the structure intake declines (falls through, no scaffold).
#[test]
fn structure_intake_declines_without_a_spec_file() {
    let root = fresh_root("noscaffold");
    let mut s = CodingAgentSession::new(&root, GuardrailPolicy::default());
    let r = s.handle_query("make a new project organized like structure.md");
    assert_ne!(r.workflow, "site.scaffold", "no spec file on disk: {}", r.response);
    let _ = fs::remove_dir_all(&root);
}

/// THE GROWTH LOOP, end to end: an unknown concept is TAUGHT at runtime and the
/// very next ask uses it — resolved through the real resolver (synonym too),
/// emitted, and request-fidelity verified. Vocabulary grows without code.
#[test]
fn teach_then_use_grows_the_web_vocabulary() {
    let reg_file = std::env::temp_dir().join(format!("nsynth_teachreg_{}.json", std::process::id()));
    let _ = fs::remove_file(&reg_file);
    std::env::set_var("NSYNTH_WEB_REGISTRY", &reg_file);
    let root = fresh_root("teach");
    let mut s = CodingAgentSession::new(&root, GuardrailPolicy::default());

    // BEFORE: "testimonials" resolves to nothing — the page builds without it.
    let r0 = s.handle_query("make a new page called home for my website with a hero and testimonials");
    assert!(r0.success);
    let html0 = fs::read_to_string(root.join("site/home.html")).unwrap();
    assert!(!html0.contains("id=\"testimonials\""), "unknown concept must not fabricate");

    // TEACH.
    let rt = s.handle_query(
        "teach web: a testimonials section means customer quotes displayed in a row, also called reviews",
    );
    assert!(rt.success, "{}", rt.response);
    assert_eq!(rt.workflow, "registry.teach");

    // AFTER: the SYNONYM ("reviews") reaches the taught concept; the section is
    // emitted and fidelity-verified on the new page.
    let r1 = s.handle_query("add a new page called landing to my website with a hero and reviews");
    assert!(r1.success, "{}", r1.response);
    let html1 = fs::read_to_string(root.join("site/landing.html")).unwrap();
    assert!(html1.contains("id=\"testimonials\""), "taught section emitted via synonym: {html1}");
    assert!(html1.contains("customer quotes"), "definition-derived content");

    std::env::remove_var("NSYNTH_WEB_REGISTRY");
    let _ = fs::remove_file(&reg_file);
    let _ = fs::remove_dir_all(&root);
}

/// BACKEND through the front door: "make me an api with a health check" builds
/// a compile-gated server with the health route; component/op asks unaffected.
#[test]
fn handle_query_builds_backend_from_prose() {
    let root = fresh_root("backend");
    let mut s = CodingAgentSession::new(&root, GuardrailPolicy::default());
    let r = s.handle_query("make me an api with a health check and a users database");
    assert!(r.success, "response: {}", r.response);
    assert_eq!(r.workflow, "backend.build");
    let src = fs::read_to_string(root.join("backend/main.rs")).expect("server on disk");
    assert!(src.contains("/health"), "health route present");
    // Guard: a counter ask still routes to components, not backend.
    let r2 = s.handle_query("build a counter");
    assert_eq!(r2.workflow, "component.build", "{}", r2.response);
    let _ = fs::remove_dir_all(&root);
}

/// SITE + BACKEND MEET: "a contact form that posts to my api" wires the form's
/// action to the generated server's route — comprehended through the hub's
/// backend domain, emitted, and fidelity-verified.
#[test]
fn handle_query_wires_contact_form_to_the_api() {
    let root = fresh_root("meet");
    let mut s = CodingAgentSession::new(&root, GuardrailPolicy::default());
    let r = s.handle_query(
        "add a new page called reach to my website with a contact form that posts to my api",
    );
    assert!(r.success, "response: {}", r.response);
    let html = fs::read_to_string(root.join("site/reach.html")).expect("page");
    assert!(
        html.contains("action=\"/rules/contact/evaluate\"") && html.contains("method=\"post\""),
        "form wired to the api: {html}"
    );
    // Without the api phrase, the form stays unwired.
    let r2 = s.handle_query("add a new page called plain to my website with a contact form");
    assert!(r2.success, "{}", r2.response);
    let html2 = fs::read_to_string(root.join("site/plain.html")).expect("page2");
    assert!(!html2.contains("action=\"/rules/"), "unrequested wiring must not appear");
    let _ = fs::remove_dir_all(&root);
}
