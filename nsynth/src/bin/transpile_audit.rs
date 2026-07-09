//! TRANSPILE SELF-AUDIT — compile-check every library op's Rust emission with real rustc.
//!
//! The Mog->Rust transpiler is only trustworthy if its output actually compiles. This audits the
//! WHOLE op library automatically: transpile each op, hand the Rust to rustc as a lib crate, and
//! report which fail + the error class. It replaces hand-probing one op at a time — the transpiler
//! validates itself. Run it as a regression guard and as a gap-finder (each failure is a worklist
//! item with its rustc error, grouped by kind so systematic gaps surface as a cluster, not a
//! one-off). Exit code = number of failures (0 = every op compiles).
use mog_synth::mog_transpile::to_rust;
use mog_synth::op_library::OPS;
use std::collections::BTreeMap;
use std::process::Command;

fn error_class(stderr: &str) -> String {
    let line = stderr
        .lines()
        .find(|l| l.contains("error[") || l.trim_start().starts_with("error"))
        .unwrap_or("")
        .trim()
        .to_string();
    // Collapse to a stable class: the error code if present, else a trimmed message.
    if let Some(code_start) = line.find("error[") {
        let rest = &line[code_start + 6..];
        if let Some(end) = rest.find(']') {
            return format!("error[{}]", &rest[..end]);
        }
    }
    line.chars().take(60).collect()
}

fn main() {
    let only = std::env::args().nth(1); // optional: audit a single op by name
    let tmp = std::env::temp_dir().join("nsynth_transpile_audit");
    let _ = std::fs::create_dir_all(&tmp);

    let mut pass = 0usize;
    let mut failures: Vec<(String, String, String)> = Vec::new(); // (name, class, full_first_errs)
    let mut by_class: BTreeMap<String, usize> = BTreeMap::new();

    for op in OPS {
        if let Some(name) = &only {
            if op.name != name {
                continue;
            }
        }
        let rust = to_rust(op.mog);
        let src = tmp.join(format!("{}.rs", op.name));
        if std::fs::write(&src, &rust).is_err() {
            continue;
        }
        let out = Command::new("rustc")
            .args([
                "--crate-type",
                "lib",
                "--edition",
                "2021",
                "-A",
                "warnings",
                "--emit",
                "metadata",
                "-o",
            ])
            .arg(tmp.join(format!("{}.rmeta", op.name)))
            .arg(&src)
            .output();
        let out = match out {
            Ok(o) => o,
            Err(e) => {
                eprintln!("rustc spawn failed: {e}");
                std::process::exit(255);
            }
        };
        if out.status.success() {
            pass += 1;
        } else {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let class = error_class(&stderr);
            *by_class.entry(class.clone()).or_default() += 1;
            let errs: String = stderr
                .lines()
                .filter(|l| l.contains("error"))
                .take(2)
                .collect::<Vec<_>>()
                .join(" | ");
            failures.push((op.name.to_string(), class, errs));
            if only.is_some() {
                println!("--- transpiled Rust for {} ---\n{rust}", op.name);
            }
        }
    }

    let total = pass + failures.len();
    println!("\n==================================================");
    println!("TRANSPILE AUDIT: {pass}/{total} library ops compile as Rust");
    println!("==================================================");
    if !by_class.is_empty() {
        println!("\nFailures by error class:");
        let mut classes: Vec<_> = by_class.iter().collect();
        classes.sort_by(|a, b| b.1.cmp(a.1));
        for (class, n) in classes {
            println!("  {n:>4}  {class}");
        }
        println!("\nPer-op failures:");
        for (name, class, errs) in &failures {
            println!("  {name:<32} {class:<22} {errs}");
        }
    }
    std::process::exit(failures.len().min(254) as i32);
}
