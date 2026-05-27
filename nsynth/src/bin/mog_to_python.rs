//! CLI wrapper: Mog source → Python.
//!
//! Reads from `--in PATH` or stdin, writes to `--out PATH` or stdout.
//! Delegates to `mog_synth::mog_transpile::to_python` — see that module
//! for the transpilation rules + test coverage.

use std::io::{Read, Write};

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let input = match arg_value(&args, "--in") {
        Some(path) => std::fs::read_to_string(&path).unwrap_or_else(|err| {
            eprintln!("[mog_to_python] cannot read {path}: {err}");
            std::process::exit(1);
        }),
        None => {
            let mut buf = String::new();
            std::io::stdin()
                .read_to_string(&mut buf)
                .expect("read stdin");
            buf
        }
    };
    let py = mog_synth::mog_transpile::to_python(&input);
    match arg_value(&args, "--out") {
        Some(path) => {
            let mut f = std::fs::File::create(&path).unwrap_or_else(|err| {
                eprintln!("[mog_to_python] cannot open {path}: {err}");
                std::process::exit(1);
            });
            let _ = f.write_all(py.as_bytes());
        }
        None => print!("{}", py),
    }
}
