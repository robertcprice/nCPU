#!/usr/bin/env bash
# Prune rebuildable build and test caches for the nCPU project.
#
# Default behavior cleans:
#   - kernels/rust_metal/target        Rust build artifacts (often ~1 GiB)
#   - **/__pycache__                   Python bytecode caches
#   - .pytest_cache                    pytest state
#   - .mypy_cache / .ruff_cache        type/lint caches
#   - *.egg-info                       setup.py metadata
#   - .coverage                        coverage.py data file
#
# Extras (opt-in):
#   --prune-previews N   Keep only the newest N paper/generated/*-preview-*
#                        directories (default: keep all)
#   --pip-cache          Also run `pip cache purge`
#   --cargo-registry     Also nuke ~/.cargo/registry/src (forces crate re-unpack)
#   --dry-run            Show what would be removed; don't delete anything
#
# Nothing here touches user-owned data (Hugging Face models, downloads,
# Library/Caches). Run those by hand if you need them gone.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

dry_run=0
prune_previews=""
clean_pip_cache=0
clean_cargo_registry=0

usage() {
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prune-previews)
      prune_previews="${2:-}"
      [[ -z "$prune_previews" ]] && { echo "--prune-previews needs a number" >&2; exit 2; }
      shift 2
      ;;
    --pip-cache)
      clean_pip_cache=1
      shift
      ;;
    --cargo-registry)
      clean_cargo_registry=1
      shift
      ;;
    --dry-run)
      dry_run=1
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

free_before=$(df -k . | awk 'NR==2 {print $4}')

remove() {
  local target="$1"
  if [[ ! -e "$target" ]]; then
    return 0
  fi
  if [[ "$dry_run" -eq 1 ]]; then
    printf "  would remove: %s\n" "$target"
  else
    rm -rf -- "$target"
    printf "  removed: %s\n" "$target"
  fi
}

echo "[clean-caches] scope: $repo_root"
[[ "$dry_run" -eq 1 ]] && echo "[clean-caches] DRY RUN — no files will be deleted"

echo "[clean-caches] project build caches"
remove "kernels/rust_metal/target"
remove ".pytest_cache"
remove ".mypy_cache"
remove ".ruff_cache"
remove ".coverage"

echo "[clean-caches] __pycache__ trees"
# -prune after matching so we don't descend into __pycache__ we're about to rm
if [[ "$dry_run" -eq 1 ]]; then
  find . -type d -name __pycache__ -print 2>/dev/null | head -50 | sed 's/^/  would remove: /'
  count=$(find . -type d -name __pycache__ 2>/dev/null | wc -l | tr -d ' ')
  echo "  ($count __pycache__ dirs total)"
else
  find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
fi

echo "[clean-caches] *.egg-info"
if [[ "$dry_run" -eq 1 ]]; then
  find . -type d -name '*.egg-info' -print 2>/dev/null | sed 's/^/  would remove: /'
else
  find . -type d -name '*.egg-info' -prune -exec rm -rf {} + 2>/dev/null || true
fi

if [[ -n "$prune_previews" ]]; then
  keep="$prune_previews"
  echo "[clean-caches] paper/generated previews (keep newest $keep)"
  if [[ -d paper/generated ]]; then
    # shellcheck disable=SC2012
    ls -1t paper/generated 2>/dev/null \
      | grep -E '(-preview-|^hotloop-preview-)' \
      | tail -n +"$((keep + 1))" \
      | while read -r d; do
          remove "paper/generated/$d"
        done
  fi
fi

if [[ "$clean_pip_cache" -eq 1 ]]; then
  echo "[clean-caches] pip cache"
  if [[ "$dry_run" -eq 1 ]]; then
    echo "  would run: pip cache purge"
  else
    pip cache purge >/dev/null 2>&1 || echo "  pip cache purge failed (pip missing?)"
  fi
fi

if [[ "$clean_cargo_registry" -eq 1 ]]; then
  echo "[clean-caches] ~/.cargo/registry/src"
  remove "$HOME/.cargo/registry/src"
fi

free_after=$(df -k . | awk 'NR==2 {print $4}')
freed_kib=$(( free_after - free_before ))
if (( freed_kib > 0 )); then
  awk -v k="$freed_kib" 'BEGIN{printf "[clean-caches] freed %.1f MiB\n", k/1024}'
else
  echo "[clean-caches] done (no measurable change; caches may have been empty)"
fi
