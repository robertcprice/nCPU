#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

paper_path="${repo_root}/paper/ncpu_paper.md"
output_path="${repo_root}/paper/ncpu_paper.pdf"
pdf_engine="${PDF_ENGINE:-lualatex}"
include_toc=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Render the paper markdown to PDF with publication-oriented defaults.

Options:
  --paper-path PATH   Source markdown file (default: paper/ncpu_paper.md)
  --output-path PATH  Output PDF path (default: paper/ncpu_paper.pdf)
  --pdf-engine NAME   Pandoc PDF engine (default: lualatex)
  --toc               Include a table of contents in the rendered PDF
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --paper-path)
      paper_path="$2"
      shift 2
      ;;
    --output-path)
      output_path="$2"
      shift 2
      ;;
    --pdf-engine)
      pdf_engine="$2"
      shift 2
      ;;
    --toc)
      include_toc=1
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

if [[ ! -f "$paper_path" ]]; then
  echo "Paper markdown not found: $paper_path" >&2
  exit 1
fi

if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc is required to build the paper PDF" >&2
  exit 1
fi

if ! command -v "$pdf_engine" >/dev/null 2>&1; then
  echo "Requested PDF engine is not available: $pdf_engine" >&2
  exit 1
fi

mkdir -p "$(dirname "$output_path")"

title="$(sed -n 's/^# //p' "$paper_path" | head -n 1)"
author="$(sed -n 's/^\*\*\(.*\)\*\*$/\1/p' "$paper_path" | head -n 1)"
paper_date="$(sed -n 's/^\*\(.*\)\*$/\1/p' "$paper_path" | head -n 1)"

pandoc_args=(
  "$paper_path"
  "--standalone"
  "--from" "gfm"
  "--pdf-engine=$pdf_engine"
  "--resource-path=${repo_root}:${repo_root}/paper"
  "-V" "geometry:margin=1in"
  "-V" "papersize=letter"
  "-V" "mainfont=STIX Two Text"
  "-V" "mainfontfallback=Arial Unicode MS:"
  "-V" "mainfontfallback=Apple Symbols:"
  "-V" "sansfont=Helvetica Neue"
  "-V" "sansfontfallback=Arial Unicode MS:"
  "-V" "sansfontfallback=Apple Symbols:"
  "-V" "monofont=Menlo"
  "-V" "monofontfallback=Arial Unicode MS:"
  "-V" "monofontfallback=Apple Symbols:"
  "-V" "mathfont=STIX Two Math"
  "-V" "CJKmainfont=Songti SC"
  "-V" "CJKsansfont=Hiragino Sans"
  "-V" "CJKmonofont=Songti SC"
  "--metadata" "title=${title}"
  "--metadata" "author=${author}"
  "--metadata" "date=${paper_date}"
  "-o" "$output_path"
)

if [[ "$include_toc" -eq 1 ]]; then
  pandoc_args+=("--toc")
fi

echo "[paper-pdf] Rendering ${paper_path} -> ${output_path}"
pandoc "${pandoc_args[@]}"

echo "[paper-pdf] Done"
