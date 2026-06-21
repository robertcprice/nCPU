//! CLI: run NL synthesis repo workflow fixtures (Package L scaffold).

use mog_synth::agent::repo::{GuardrailPolicy, RepoWorkflowRunner};
use std::env;
use std::path::PathBuf;
use std::process;

fn usage() {
    eprintln!(
        "usage: repo_workflow --root <repo_path> [--fixtures] [--report <name>]\n\
          --fixtures   run nl_synthesis_fixture_suite tasks (default)\n\
          --report     save WorkflowRunReport JSON under .nsynth/workflows/"
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let root = args
        .iter()
        .position(|a| a == "--root")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from);
    let run_fixtures = !args.iter().any(|a| a == "--no-fixtures");
    let report_name = args
        .iter()
        .position(|a| a == "--report")
        .and_then(|i| args.get(i + 1).cloned())
        .unwrap_or_else(|| "workflow_run".to_string());

    let Some(root) = root else {
        usage();
        process::exit(2);
    };

    if !root.is_dir() {
        eprintln!("error: --root is not a directory: {}", root.display());
        process::exit(2);
    }

    let mut runner = RepoWorkflowRunner::new(&root, GuardrailPolicy::default());
    let report = if run_fixtures {
        runner.run_nl_fixtures()
    } else {
        eprintln!("error: no workflow mode selected (use --fixtures)");
        process::exit(2);
    };

    let saved = runner
        .save_report(&report, &report_name)
        .unwrap_or_else(|e| {
            eprintln!("warning: could not save workflow report: {e}");
            PathBuf::from("")
        });

    let report_path = if saved.file_name().is_some() {
        Some(saved.to_string_lossy().to_string())
    } else {
        None
    };
    let summary = serde_json::json!({
        "root": report.root,
        "total": report.total,
        "succeeded": report.succeeded,
        "report_path": report_path,
        "outcomes": report.outcomes,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&summary).unwrap_or_default()
    );

    if report.succeeded != report.total {
        process::exit(1);
    }
}
