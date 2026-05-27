#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

python_bin="${PYTHON:-python3}"
quick_mode=0
bootstrap_mog=1
run_full_pytest=1

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Maintainer verification for publication/release readiness.

Options:
  --quick               Run a fast smoke pass instead of the full suite
  --skip-mog-bootstrap  Do not clone/build the external Mog toolchain
  --skip-full-pytest    Skip the full pytest pass
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      quick_mode=1
      run_full_pytest=0
      shift
      ;;
    --skip-mog-bootstrap)
      bootstrap_mog=0
      shift
      ;;
    --skip-full-pytest)
      run_full_pytest=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$repo_root"

echo "[release-check] nCPU doctor"
"$python_bin" -m ncpu doctor

if [[ "$bootstrap_mog" -eq 1 ]]; then
  echo "[release-check] Bootstrapping Mog toolchain"
  "$repo_root/scripts/setup/bootstrap_mog_toolchain.sh"
fi

echo "[release-check] Building wheel"
"$python_bin" -m pip wheel . --no-deps -w "${TMPDIR:-/tmp}/ncpu-wheel-check"

echo "[release-check] Running focused smoke tests"
"$python_bin" -m pytest \
  tests/test_cli_entrypoints.py \
  tests/test_package_metadata.py \
  tests/mog/test_mog_execute_errors.py \
  -q

if command -v cargo >/dev/null 2>&1 && [[ "$(uname -s)" == "Darwin" ]]; then
  echo "[release-check] cargo check (rust_metal)"
  cargo check --manifest-path kernels/rust_metal/Cargo.toml --bin ncpu_run

  if "$python_bin" -c "from benchmarks.benchmark_gpu_only import load_ncpu_metal; import sys; sys.exit(0 if load_ncpu_metal() is not None else 1)" 2>/dev/null; then
    echo "[release-check] Strict Rust/Metal GPU-only smoke (--rust-only, counted workload)"
    smoke_dir="$(mktemp -d)"
    if "$python_bin" benchmarks/export_gpu_only_matrix.py \
        --output-dir "$smoke_dir" \
        --rust-only \
        --workload counted \
        --timeout-seconds 60; then
      rm -rf "$smoke_dir"
    else
      echo "[release-check] Strict Rust/Metal smoke failed; preserving artifacts at $smoke_dir" >&2
      exit 1
    fi
  else
    echo "[release-check] Skipping strict Rust/Metal smoke (ncpu_metal not importable)"
  fi
else
  echo "[release-check] Skipping cargo check and Rust/Metal smoke (requires cargo + Darwin maintainer environment)"
fi

if [[ "$quick_mode" -eq 1 ]]; then
  echo "[release-check] Quick mode complete"
  exit 0
fi

if [[ "$run_full_pytest" -eq 1 ]]; then
  echo "[release-check] Running full pytest suite"
  "$python_bin" -m pytest tests/ -q
fi

echo "[release-check] Complete"
