#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

artifact_dir=""
baseline_link="${repo_root}/paper/generated/baseline"

usage() {
  cat <<EOF
Usage: $(basename "$0") --artifact-dir PATH [options]

Point the canonical publication baseline link at a verified artifact bundle.

Options:
  --artifact-dir PATH    Verified artifact directory to promote
  --baseline-link PATH   Symlink path to update (default: paper/generated/baseline)
  -h, --help             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifact-dir)
      artifact_dir="$2"
      shift 2
      ;;
    --baseline-link)
      baseline_link="$2"
      shift 2
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

if [[ -z "$artifact_dir" ]]; then
  echo "--artifact-dir is required" >&2
  usage >&2
  exit 2
fi

artifact_dir_abs="$(cd "$artifact_dir" && pwd)"
baseline_parent="$(cd "$(dirname "$baseline_link")" && pwd)"
baseline_name="$(basename "$baseline_link")"

required_files=(
  "${artifact_dir_abs}/artifact_manifest.json"
  "${artifact_dir_abs}/benchmarks/baseline_comparison_results.json"
  "${artifact_dir_abs}/benchmarks/ablation_results.json"
  "${artifact_dir_abs}/benchmarks/real_workload_results.json"
  "${artifact_dir_abs}/paper_tables/paper_metrics.json"
)

for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Artifact bundle is missing required file: $path" >&2
    exit 1
  fi
done

mkdir -p "$baseline_parent"
relative_target="$(
  ARTIFACT_DIR_ABS="$artifact_dir_abs" BASELINE_PARENT="$baseline_parent" python3 - <<'PY'
import os
print(os.path.relpath(os.environ["ARTIFACT_DIR_ABS"], os.environ["BASELINE_PARENT"]))
PY
)"

(
  cd "$baseline_parent"
  ln -sfn "$relative_target" "$baseline_name"
)

echo "[baseline] Promoted publication baseline"
echo "  Baseline link: ${baseline_parent}/${baseline_name}"
echo "  Target: ${artifact_dir_abs}"
