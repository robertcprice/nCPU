#!/usr/bin/env bash
# Build publication PDFs from the markdown sources.
#
#   ./paper/build_pdf.sh            # main paper -> paper/generated/ncpu_paper.pdf
#   ./paper/build_pdf.sh --all      # main paper + every companion section
#
# Requires pandoc + xelatex (both checked below). Output goes to
# paper/generated/ so source dirs stay clean; generated/ is gitignored.

set -euo pipefail
cd "$(dirname "$0")"

command -v pandoc >/dev/null || { echo "pandoc not found (brew install pandoc)"; exit 1; }
command -v xelatex >/dev/null || { echo "xelatex not found (install MacTeX/BasicTeX)"; exit 1; }

mkdir -p generated

PANDOC_FLAGS=(
  --pdf-engine=xelatex
  --toc --toc-depth=2
  --number-sections
  -V geometry:margin=1in
  -V fontsize=11pt
  # macOS system fonts: Times has Greek; Menlo has box-drawing + Greek for
  # the ASCII architecture diagrams. Latin Modern lacks both.
  -V mainfont="Times New Roman"
  -V monofont="Menlo"
  -V monofontoptions=Scale=0.82
  -V linkcolor=blue
  -V documentclass=article
  --highlight-style=tango
)

build() {
  local src="$1" out="$2"
  echo "==> $src -> $out"
  pandoc "$src" "${PANDOC_FLAGS[@]}" -o "$out"
}

build ncpu_paper.md generated/ncpu_paper.pdf

# arXiv refuses TeX-produced PDFs without source; emit the .tex so the
# submission can upload source + let arXiv compile.
echo "==> ncpu_paper.md -> generated/ncpu_paper.tex"
pandoc ncpu_paper.md "${PANDOC_FLAGS[@]}" -s -o generated/ncpu_paper.tex

if [[ "${1:-}" == "--all" ]]; then
  for f in sections/*.md; do
    base="$(basename "$f" .md)"
    build "$f" "generated/${base}.pdf"
  done
fi

echo "done. PDFs in paper/generated/"
