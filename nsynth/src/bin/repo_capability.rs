//! REAL repo-synthesis capability: for each NL fixture, scaffold a Rust crate whose
//! target function has a WRONG stub + a cargo test that fails, then run the repo agent
//! (localize -> synthesize -> repair -> cargo-test oracle -> promote) and report whether
//! it made the crate's tests pass. Unlike repo_workflow --fixtures (which needs a
//! pre-scaffolded root), this scaffolds every fixture, so it measures the actual
//! end-to-end repo capability with a real compiler+test oracle.
//!
//!   cargo run --release --bin repo_capability
use mog_synth::agent::coding_intent::CodingIntent;
use mog_synth::agent::repo::{
    nl_fixture_cargo_test_command, write_nl_fixture_crate, GuardrailPolicy, RepoAgent,
};
use mog_synth::agent::runtime::CodeTaskSpec;

/// (fixture id, NL issue the user would type). Issue must let CodingIntent resolve the
/// target and match the fixture's failing function.
const FIXTURES: &[(&str, &str)] = &[
    ("nl_fixture_add", "add two numbers"),
    ("nl_fixture_subtract", "subtract two numbers"),
    ("nl_fixture_multiply", "multiply two numbers"),
    ("nl_fixture_divide", "divide two numbers"),
    ("nl_fixture_max", "maximum of two numbers"),
    ("nl_fixture_negate", "negate a number"),
    ("nl_fixture_triple", "triple a number"),
    ("nl_fixture_square", "square a number"),
    ("nl_fixture_abs", "absolute value of a number"),
    ("nl_fixture_gcd", "greatest common divisor"),
    ("nl_fixture_arrsum", "sum of a list"),
    ("nl_fixture_arrmax", "maximum of a list"),
    ("nl_fixture_arrlen", "length of a list"),
    ("nl_fixture_reverse", "reverse a list"),
    ("nl_fixture_multifile_multiply", "multiply two numbers"),
];

/// All `.rs` files under `dir`, recursively (small, for the --dump diagnostic).
fn walkdir_rs(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else { return out };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            out.extend(walkdir_rs(&p));
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
    out
}

fn main() {
    let budget_ms: u64 = std::env::var("NSYNTH_QUERY_BUDGET_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30000);

    // Debug: `repo_capability --dump <fixture>` scaffolds one fixture, runs the agent,
    // and prints every synthesized src file + the run error, to diagnose a failure.
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--dump") {
        let fixture = args.get(i + 1).cloned().unwrap_or_default();
        let issue = FIXTURES
            .iter()
            .find(|(f, _)| *f == fixture)
            .map(|(_, is)| *is)
            .unwrap_or("fix it");
        let root = std::env::temp_dir().join(format!("nsynth_dump_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        write_nl_fixture_crate(&root, &fixture).expect("scaffold");
        let cmd = nl_fixture_cargo_test_command(&fixture).expect("cmd");
        let intent = CodingIntent::from_nl(issue).expect("intent");
        let mut spec =
            CodeTaskSpec::from_nl(root.to_string_lossy(), issue, intent, cmd, vec!["src/**".into()], 3);
        spec.budget.max_wall_ms = budget_ms;
        let mut agent = RepoAgent::new(&root, GuardrailPolicy::default());
        let result = agent.run(&spec);
        println!("issue={issue:?} baseline={} final={} success={}\nerror={:?}\nphases={:?}\n",
            result.baseline_passed, result.final_passed, result.success, result.error, result.phases_completed);
        for entry in walkdir_rs(&root.join("src")) {
            if let Ok(body) = std::fs::read_to_string(&entry) {
                println!("===== {} =====\n{body}", entry.display());
            }
        }
        let _ = std::fs::remove_dir_all(&root);
        return;
    }
    let base = std::env::temp_dir().join(format!("nsynth_repo_cap_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);

    let (mut solved, mut baseline_wrong, mut total) = (0usize, 0usize, 0usize);
    println!("{:<34} {:>8} {:>8} {:>8}   {}", "FIXTURE", "baseline", "final", "success", "note");
    for &(fixture, issue) in FIXTURES {
        total += 1;
        let root = base.join(fixture);
        let _ = std::fs::remove_dir_all(&root);
        if let Err(e) = write_nl_fixture_crate(&root, fixture) {
            println!("{fixture:<34} scaffold error: {e}");
            continue;
        }
        let Some(cmd) = nl_fixture_cargo_test_command(fixture) else {
            println!("{fixture:<34} no cargo-test command");
            continue;
        };
        let intent = match CodingIntent::from_nl(issue) {
            Ok(i) => i,
            Err(e) => {
                println!("{fixture:<34} intent-refused: {e:?}   (\"{issue}\")");
                continue;
            }
        };
        let spec = CodeTaskSpec::from_nl(
            root.to_string_lossy(),
            issue,
            intent,
            cmd,
            vec!["src/**".into()],
            3,
        );
        let mut agent = RepoAgent::new(&root, GuardrailPolicy::default());
        // Per-task wall bound (the fix from this session); the agent also honors it.
        let mut spec = spec;
        spec.budget.max_wall_ms = budget_ms;
        let result = agent.run(&spec);
        if !result.baseline_passed {
            baseline_wrong += 1;
        }
        if result.success && result.final_passed {
            solved += 1;
        }
        let note = result.error.clone().unwrap_or_default();
        println!(
            "{fixture:<34} {:>8} {:>8} {:>8}   {}",
            result.baseline_passed,
            result.final_passed,
            result.success,
            &note[..note.len().min(60)]
        );
        let _ = std::fs::remove_dir_all(&root);
    }
    let _ = std::fs::remove_dir_all(&base);
    println!(
        "\nREPO CAPABILITY: {solved}/{total} solved end-to-end (cargo-test oracle) | \
         {baseline_wrong}/{total} started with a failing baseline (as intended)"
    );
}
