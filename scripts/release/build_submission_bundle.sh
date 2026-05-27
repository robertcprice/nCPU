#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

python_bin="${PYTHON:-python3}"
artifact_dir=""
paper_path="${repo_root}/paper/ncpu_paper.md"
pdf_path=""
output_dir=""
output_zip=""
skip_pdf_build=0

usage() {
  cat <<EOF
Usage: $(basename "$0") --artifact-dir PATH [options]

Assemble a publication submission bundle from a verified artifact run.

Options:
  --artifact-dir PATH   Artifact directory produced by paper_artifacts.sh
  --paper-path PATH     Paper markdown file (default: paper/ncpu_paper.md)
  --pdf-path PATH       Existing paper PDF to include
  --output-dir PATH     Bundle directory (default: ARTIFACT_DIR/submission-bundle)
  --output-zip PATH     Optional zip path (default: OUTPUT_DIR.zip)
  --skip-pdf-build      Reuse --pdf-path and do not rebuild the PDF
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifact-dir)
      artifact_dir="$2"
      shift 2
      ;;
    --paper-path)
      paper_path="$2"
      shift 2
      ;;
    --pdf-path)
      pdf_path="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --output-zip)
      output_zip="$2"
      shift 2
      ;;
    --skip-pdf-build)
      skip_pdf_build=1
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

if [[ -z "$artifact_dir" ]]; then
  echo "--artifact-dir is required" >&2
  usage >&2
  exit 2
fi

if [[ ! -d "$artifact_dir" ]]; then
  echo "Artifact directory not found: $artifact_dir" >&2
  exit 1
fi

if [[ ! -f "$paper_path" ]]; then
  echo "Paper markdown not found: $paper_path" >&2
  exit 1
fi

if [[ -z "$output_dir" ]]; then
  output_dir="${artifact_dir}/submission-bundle"
fi

if [[ -z "$output_zip" ]]; then
  output_zip="${output_dir}.zip"
fi

mkdir -p "$output_dir" "$output_dir/benchmarks" "$output_dir/paper_tables"

bundle_paper_md="${output_dir}/$(basename "$paper_path")"
bundle_pdf_path="${output_dir}/$(basename "${paper_path%.md}.pdf")"
preview_path="${output_dir}/paper_claims_preview.md"

echo "[submission-bundle] Verifying paper claims"
"$python_bin" "${repo_root}/scripts/release/verify_paper_claims.py" \
  --artifact-dir "$artifact_dir" \
  --paper-path "$paper_path" \
  --output-path "$preview_path"

echo "[submission-bundle] Copying paper markdown"
cp -f "$paper_path" "$bundle_paper_md"

if [[ "$skip_pdf_build" -eq 1 ]]; then
  if [[ -z "$pdf_path" ]]; then
    echo "--skip-pdf-build requires --pdf-path" >&2
    exit 2
  fi
  if [[ ! -f "$pdf_path" ]]; then
    echo "Paper PDF not found: $pdf_path" >&2
    exit 1
  fi
  echo "[submission-bundle] Reusing existing PDF"
  cp -f "$pdf_path" "$bundle_pdf_path"
else
  echo "[submission-bundle] Rendering paper PDF"
  "${repo_root}/scripts/release/build_paper_pdf.sh" \
    --paper-path "$paper_path" \
    --output-path "$bundle_pdf_path"
fi

copied_files=(
  "$(basename "$bundle_paper_md")"
  "$(basename "$bundle_pdf_path")"
  "$(basename "$preview_path")"
)

copy_optional_top_level() {
  local pattern
  for pattern in "$@"; do
    local source
    for source in "${artifact_dir}"/${pattern}; do
      if [[ ! -f "$source" ]]; then
        continue
      fi
      cp -f "$source" "${output_dir}/"
      copied_files+=("$(basename "$source")")
    done
  done
}

copy_tree() {
  local source_dir="$1"
  local target_subdir="$2"
  if [[ ! -d "$source_dir" ]]; then
    return
  fi

  local source
  while IFS= read -r source; do
    local rel_path
    rel_path="${source#"${source_dir}/"}"
    mkdir -p "$(dirname "${output_dir}/${target_subdir}/${rel_path}")"
    cp -f "$source" "${output_dir}/${target_subdir}/${rel_path}"
    copied_files+=("${target_subdir}/${rel_path}")
  done < <(find "$source_dir" -type f | sort)
}

copy_tree_to_exact_path() {
  local source_dir="$1"
  local target_dir="$2"
  if [[ ! -d "$source_dir" ]]; then
    return
  fi

  local source
  while IFS= read -r source; do
    local rel_path
    rel_path="${source#"${source_dir}/"}"
    mkdir -p "$(dirname "${output_dir}/${target_dir}/${rel_path}")"
    cp -f "$source" "${output_dir}/${target_dir}/${rel_path}"
    copied_files+=("${target_dir}/${rel_path}")
  done < <(find "$source_dir" -type f | sort)
}

copy_optional_top_level \
  "artifact_manifest.json" \
  "doctor.txt" \
  "comparison*.json" \
  "comparison*.md"

copy_tree "${artifact_dir}/benchmarks" "benchmarks"
copy_tree "${artifact_dir}/paper_tables" "paper_tables"
copy_tree_to_exact_path "${repo_root}/paper/generated/meta_comparison_demo_latest" "paper/generated/meta_comparison_demo_latest"

echo "[submission-bundle] Writing SHA256SUMS.txt"
(
  cd "$output_dir"
  shasum -a 256 "${copied_files[@]}" > SHA256SUMS.txt
)

mkdir -p "$(dirname "$output_zip")"
output_zip_abs="$(cd "$(dirname "$output_zip")" && pwd)/$(basename "$output_zip")"

if command -v zip >/dev/null 2>&1; then
  echo "[submission-bundle] Writing ${output_zip_abs}"
  rm -f "$output_zip_abs"
  (
    cd "$(dirname "$output_dir")"
    zip -qr "$output_zip_abs" "$(basename "$output_dir")"
  )
else
  echo "[submission-bundle] zip not found; bundle directory left at ${output_dir}"
fi

echo "[submission-bundle] Done"
echo "  Bundle directory: ${output_dir}"
if [[ -f "$output_zip_abs" ]]; then
  echo "  Bundle zip: ${output_zip_abs}"
fi
