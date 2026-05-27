#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

python_bin="${PYTHON:-python3}"
quick_mode=0
bootstrap_mog=1
run_baseline=1
run_ablation=1
run_real_workload=1
trials=5
output_dir="${repo_root}/paper/generated/$(date -u +%Y%m%dT%H%M%SZ)"
baseline_dir=""
compare_mode="auto"
verify_paper_claims=0
paper_path="${repo_root}/paper/ncpu_paper.md"
build_paper_pdf=0
build_submission_bundle=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Generate publication-oriented benchmark artifacts under paper/generated/.

Options:
  --quick               Fast path: 1 trial, reduced real-workload scope
  --trials N            Number of trials for the benchmark scripts (default: 5)
  --output-dir PATH     Target artifact directory
  --baseline-dir PATH   Optional prior artifact directory to compare against
  --verify-paper-claims Fail if the paper's measured benchmark sections drift from this artifact run
  --paper-path PATH     Paper markdown file for claim verification (default: paper/ncpu_paper.md)
  --build-paper-pdf     Render a fresh PDF for the paper into the artifact directory
  --build-submission-bundle
                        Assemble a publication bundle under OUTPUT_DIR/submission-bundle
  --strict-compare      Fail the run if artifact comparison reports regressions
  --informational-compare
                        Always write comparison reports but never fail the run on comparison
  --skip-mog-bootstrap  Do not clone/build the external Mog toolchain
  --skip-baseline       Skip baseline_comparison.py
  --skip-ablation       Skip ablation_study.py
  --skip-real-workload  Skip benchmark_real_workload.py
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)
      quick_mode=1
      trials=1
      shift
      ;;
    --trials)
      trials="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --baseline-dir)
      baseline_dir="$2"
      shift 2
      ;;
    --verify-paper-claims)
      verify_paper_claims=1
      shift
      ;;
    --paper-path)
      paper_path="$2"
      shift 2
      ;;
    --build-paper-pdf)
      build_paper_pdf=1
      shift
      ;;
    --build-submission-bundle)
      build_submission_bundle=1
      shift
      ;;
    --strict-compare)
      compare_mode="strict"
      shift
      ;;
    --informational-compare)
      compare_mode="informational"
      shift
      ;;
    --skip-mog-bootstrap)
      bootstrap_mog=0
      shift
      ;;
    --skip-baseline)
      run_baseline=0
      shift
      ;;
    --skip-ablation)
      run_ablation=0
      shift
      ;;
    --skip-real-workload)
      run_real_workload=0
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

mkdir -p "$output_dir/benchmarks"

if [[ "$bootstrap_mog" -eq 1 ]]; then
  echo "[paper-artifacts] Bootstrapping Mog toolchain"
  "$repo_root/scripts/setup/bootstrap_mog_toolchain.sh"
fi

echo "[paper-artifacts] Capturing environment doctor"
"$python_bin" -m ncpu doctor | tee "$output_dir/doctor.txt"

if [[ "$run_baseline" -eq 1 ]]; then
  echo "[paper-artifacts] Running baseline comparison"
  "$python_bin" benchmarks/baseline_comparison.py \
    --trials "$trials" \
    --output-dir "$output_dir/benchmarks"
fi

if [[ "$run_ablation" -eq 1 ]]; then
  echo "[paper-artifacts] Running ablation study"
  "$python_bin" benchmarks/ablation_study.py \
    --trials "$trials" \
    --output-dir "$output_dir/benchmarks"
fi

if [[ "$run_real_workload" -eq 1 ]]; then
  echo "[paper-artifacts] Running real workload benchmark"
  real_args=(
    benchmarks/benchmark_real_workload.py
    --trials "$trials"
    --output-dir "$output_dir/benchmarks"
    --quiet
  )
  if [[ "$quick_mode" -eq 1 ]]; then
    real_args+=(--programs fibonacci)
  fi
  "$python_bin" "${real_args[@]}"
fi

echo "[paper-artifacts] Exporting GPU-only hotloop matrix"
"$python_bin" benchmarks/export_gpu_only_matrix.py \
  --output-dir "$output_dir/benchmarks" \
  --rust-only \
  --no-inprocess \
  --timeout-seconds 120 \
  --resume

echo "[paper-artifacts] Running neural-vs-Meta comparison demo benchmark"
"$python_bin" benchmarks/benchmark_meta_comparison_demo.py \
  --output-dir "$output_dir/benchmarks/meta_comparison_demo"

echo "[paper-artifacts] Exporting stable meta comparison paper figure"
"$python_bin" "$repo_root/scripts/release/export_meta_comparison_figure.py" \
  --source-dir "$output_dir/benchmarks/meta_comparison_demo" \
  --output-dir "$repo_root/paper/generated/meta_comparison_demo_latest"

echo "[paper-artifacts] Building artifact manifest"
"$python_bin" "$repo_root/scripts/release/build_artifact_manifest.py" \
  --output-dir "$output_dir" \
  --label "paper-artifacts"

echo "[paper-artifacts] Extracting paper-ready tables"
"$python_bin" "$repo_root/scripts/release/extract_paper_tables.py" \
  --artifact-dir "$output_dir"

if [[ "$verify_paper_claims" -eq 1 ]]; then
  echo "[paper-artifacts] Verifying paper benchmark claims"
  "$python_bin" "$repo_root/scripts/release/verify_paper_claims.py" \
    --artifact-dir "$output_dir" \
    --paper-path "$paper_path" \
    --output-path "$output_dir/paper_claims_preview.md"
fi

if [[ -n "$baseline_dir" ]]; then
  echo "[paper-artifacts] Comparing against baseline artifacts"
  if ! "$python_bin" "$repo_root/scripts/release/compare_artifacts.py" \
    --baseline-dir "$baseline_dir" \
    --candidate-dir "$output_dir" \
    --json-output "$output_dir/comparison_report.json" \
    --markdown-output "$output_dir/comparison_report.md"; then
    if [[ "$compare_mode" == "informational" ]] || [[ "$compare_mode" == "auto" && "$quick_mode" -eq 1 ]]; then
      echo "[paper-artifacts] Comparison reported regressions; keeping quick/informational run non-gating"
    else
      exit 1
    fi
  fi
fi

rendered_pdf_path="$output_dir/$(basename "${paper_path%.md}.pdf")"

if [[ "$build_paper_pdf" -eq 1 ]]; then
  echo "[paper-artifacts] Rendering paper PDF"
  "$repo_root/scripts/release/build_paper_pdf.sh" \
    --paper-path "$paper_path" \
    --output-path "$rendered_pdf_path"
fi

if [[ "$build_submission_bundle" -eq 1 ]]; then
  echo "[paper-artifacts] Building submission bundle"
  bundle_args=(
    --artifact-dir "$output_dir"
    --paper-path "$paper_path"
    --output-dir "$output_dir/submission-bundle"
  )
  if [[ "$build_paper_pdf" -eq 1 ]]; then
    bundle_args+=(--skip-pdf-build --pdf-path "$rendered_pdf_path")
  fi
  "$repo_root/scripts/release/build_submission_bundle.sh" "${bundle_args[@]}"
fi

echo "[paper-artifacts] Done"
echo "  Output directory: $output_dir"
