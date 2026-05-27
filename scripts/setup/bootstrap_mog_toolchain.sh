#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
default_mog_root="$(cd "${repo_root}/.." && pwd)/mog"

mog_root="${MOG_ROOT:-$default_mog_root}"
mog_git_url="${MOG_GIT_URL:-https://github.com/voltropy/mog.git}"
mog_git_ref="${MOG_GIT_REF:-}"
clone_if_missing=1
build_if_present=1

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Bootstrap the external Mog compiler/runtime used by compiler-backed MOG tests.

Options:
  --mog-root PATH     Target checkout/build path (default: ${default_mog_root})
  --git-url URL       Repository to clone when missing (default: ${mog_git_url})
  --git-ref REF       Optional git ref to checkout after clone/fetch
  --skip-clone        Fail instead of cloning when the repo is missing
  --clone-only        Clone/fetch only; do not build
  --build-only        Build existing checkout only; do not clone
  -h, --help          Show this help

Environment overrides:
  MOG_ROOT, MOG_GIT_URL, MOG_GIT_REF
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mog-root)
      mog_root="$2"
      shift 2
      ;;
    --git-url)
      mog_git_url="$2"
      shift 2
      ;;
    --git-ref)
      mog_git_ref="$2"
      shift 2
      ;;
    --skip-clone)
      clone_if_missing=0
      shift
      ;;
    --clone-only)
      build_if_present=0
      shift
      ;;
    --build-only)
      clone_if_missing=0
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

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo is required to build the Mog toolchain" >&2
  exit 1
fi

if [[ ! -d "$mog_root/.git" ]]; then
  if [[ "$clone_if_missing" -ne 1 ]]; then
    echo "Mog checkout not found at $mog_root" >&2
    exit 1
  fi
  echo "[bootstrap] Cloning Mog into $mog_root"
  mkdir -p "$(dirname "$mog_root")"
  git clone "$mog_git_url" "$mog_root"
else
  echo "[bootstrap] Using existing Mog checkout at $mog_root"
fi

if [[ -n "$mog_git_ref" ]]; then
  echo "[bootstrap] Checking out Mog ref $mog_git_ref"
  git -C "$mog_root" fetch --tags --prune origin
  git -C "$mog_root" checkout "$mog_git_ref"
fi

if [[ "$build_if_present" -eq 1 ]]; then
  echo "[bootstrap] Building Mog compiler"
  cargo build --release --manifest-path "$mog_root/compiler/Cargo.toml"

  echo "[bootstrap] Building Mog runtime"
  cargo build --release --manifest-path "$mog_root/runtime-rs/Cargo.toml"
fi

mogc_path="$mog_root/compiler/target/release/mogc"
runtime_path="$mog_root/runtime-rs/target/release/libmog_runtime.a"

echo
echo "[bootstrap] Mog toolchain ready"
echo "  MOG_ROOT=$mog_root"
echo "  MOGC_BINARY=$mogc_path"
echo "  MOG_RUNTIME=$runtime_path"
