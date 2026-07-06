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

    // BEFORE: "newsletter" resolves to nothing — the page builds without it.
    let r0 = s.handle_query("make a new page called home for my website with a hero and a newsletter");
    assert!(r0.success);
    let html0 = fs::read_to_string(root.join("site/home.html")).unwrap();
    assert!(!html0.contains("id=\"newsletter\""), "unknown concept must not fabricate");

    // TEACH.
    let rt = s.handle_query(
        "teach web: a newsletter section means an email signup form in a strip, also called subscribe",
    );
    assert!(rt.success, "{}", rt.response);
    assert_eq!(rt.workflow, "registry.teach");

    // AFTER: the SYNONYM ("subscribe") reaches the taught concept; the section is
    // emitted and fidelity-verified on the new page.
    let r1 = s.handle_query("add a new page called promo to my website with a hero and subscribe");
    assert!(r1.success, "{}", r1.response);
    let html1 = fs::read_to_string(root.join("site/promo.html")).unwrap();
    assert!(html1.contains("id=\"newsletter\""), "taught section emitted via synonym: {html1}");
    assert!(html1.contains("email signup"), "definition-derived content");

    std::env::remove_var("NSYNTH_WEB_REGISTRY");
    let _ = fs::remove_file(&reg_file);
    let _ = fs::remove_dir_all(&root);
}

/// SELF-LEARNING ARCHETYPE through the front door: teach a NEW page archetype
/// by prose, then an ABSTRACT prompt that only names the purpose composes the
/// taught structure — the full "understand better prompts, make complex programs
/// from abstract asks" loop, reachable in conversation, no code change.
#[test]
fn teach_archetype_then_abstract_prompt_composes_it() {
    let reg_file = std::env::temp_dir().join(format!("nsynth_archreg_{}.json", std::process::id()));
    let _ = fs::remove_file(&reg_file);
    std::env::set_var("NSYNTH_WEB_REGISTRY", &reg_file);
    let root = fresh_root("archteach");
    let mut s = CodingAgentSession::new(&root, GuardrailPolicy::default());

    // TEACH a new archetype by prose (its parts are named in the definition).
    // ("portal" is a novel word, not a built-in archetype like dashboard/saas.)
    let rt = s.handle_query(
        "teach web: a portal archetype means a page with a hero, features, and an about story",
    );
    assert!(rt.success, "{}", rt.response);
    assert_eq!(rt.workflow, "registry.teach");

    // ABSTRACT prompt — names the purpose (archetype) but NO section words. The
    // taught archetype composes hero + features + about, emitted + fidelity-verified.
    let r = s.handle_query("build a portal page called board for my team");
    assert!(r.success, "{}", r.response);
    let html = fs::read_to_string(root.join("site/board.html")).unwrap();
    for marker in ["class=\"hero\"", "class=\"features\"", "class=\"about\""] {
        assert!(html.contains(marker), "composed section {marker} present: {html}");
    }

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

/// SITE + BACKEND MEET, CLOSED LOOP: "a contact form that posts to my api"
/// (1) wires the form's action to POST /events (comprehended through the
/// hub's backend domain, emitted, fidelity-verified), (2) PROVISIONS a real
/// structural backend in the same action when none exists (compile+serve
/// gated), and (3) the promise is smoke-proven end to end: boot the
/// provisioned server, submit the form body, see the stored submission.
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
        html.contains("action=\"/events\"") && html.contains("method=\"post\""),
        "form wired to the api: {html}"
    );
    // The wired target is REAL: the same ask provisioned a backend...
    let backend = fs::read_to_string(root.join("backend/main.rs")).expect("provisioned backend");
    // ...and the loop closes live: boot it, POST the form body, see it stored.
    let (src, bin) =
        mog_synth::backend_http::compile_to_temp_bin(&backend, false).expect("compile backend");
    let smoke = mog_synth::backend_http::verify_submission_intake(&bin, 2);
    mog_synth::backend_http::cleanup_temp_artifacts(&src, &bin);
    smoke.expect("form submission accepted and stored by the provisioned backend");
    // Without the api phrase, the form stays unwired and nothing is provisioned.
    let r2 = s.handle_query("add a new page called plain to my website with a contact form");
    assert!(r2.success, "{}", r2.response);
    let html2 = fs::read_to_string(root.join("site/plain.html")).expect("page2");
    assert!(!html2.contains("action=\"/events\""), "unrequested wiring must not appear");
    let _ = fs::remove_dir_all(&root);
}

/// MULTI-PAGE served, nav resolves over HTTP: two prompts build a two-page
/// site (nav rewired site-wide), the provisioned backend serves BOTH pages,
/// and the index's nav link to the second page is a real 200 over HTTP — the
/// inter-page navigation actually works when served, not just in the files.
#[test]
fn provisioned_backend_serves_every_page_with_working_nav() {
    let root = fresh_root("multipage");
    let mut s = CodingAgentSession::new(&root, GuardrailPolicy::default());
    // Page 1 provisions the backend (api-wired) and creates index.html.
    let r1 = s.handle_query(
        "add a new page called index to my website with a hero and a contact form that posts to my api",
    );
    assert!(r1.success, "page1: {}", r1.response);
    // Page 2 extends the site; nav is rewired in every page to include both.
    let r2 = s.handle_query("add a new page called about to my website with an about section");
    assert!(r2.success, "page2: {}", r2.response);
    let index = fs::read_to_string(root.join("site/index.html")).expect("index");
    assert!(index.contains("href=\"about.html\""), "index nav links to page 2: {index}");

    let backend = fs::read_to_string(root.join("backend/main.rs")).expect("provisioned backend");
    let (src, bin) =
        mog_synth::backend_http::compile_to_temp_bin(&backend, false).expect("compile backend");
    // Both pages serve over HTTP; the index carries the working nav link.
    let served = mog_synth::backend_http::verify_static_pages(
        &bin,
        &root.join("site"),
        &[("/", "href=\"about.html\""), ("/about.html", "<title")],
        2,
    );
    mog_synth::backend_http::cleanup_temp_artifacts(&src, &bin);
    served.expect("both pages served over HTTP with the nav link resolving");
    let _ = fs::remove_dir_all(&root);
}

/// SINGLE ARTIFACT: the provisioned backend serves the generated SITE and its
/// api from one binary. One prompt yields both; booting the backend with
/// `--static <site>` serves the page over HTTP (text/html) while /health still
/// answers. This is the full stack from one sentence, end-to-end proven.
#[test]
fn provisioned_backend_serves_the_generated_site() {
    let root = fresh_root("serve");
    let mut s = CodingAgentSession::new(&root, GuardrailPolicy::default());
    let r = s.handle_query(
        "add a new page called index to my website with a hero and a contact form that posts to my api",
    );
    assert!(r.success, "response: {}", r.response);
    // Both artifacts exist from the one ask.
    assert!(root.join("site/index.html").exists(), "site page");
    let backend = fs::read_to_string(root.join("backend/main.rs")).expect("provisioned backend");
    // Compile the provisioned backend, boot it pointed at the generated site,
    // and require it to serve the page over HTTP while the api stays live.
    let (src, bin) =
        mog_synth::backend_http::compile_to_temp_bin(&backend, false).expect("compile backend");
    let served = mog_synth::backend_http::verify_static_serving(&bin, &root.join("site"), "<title", 2);
    mog_synth::backend_http::cleanup_temp_artifacts(&src, &bin);
    served.expect("backend serves the generated site over HTTP with the api still answering");
    let _ = fs::remove_dir_all(&root);
}
