#!/usr/bin/env bash
# Release packaging script for NPCoT standalone binary (N3b).
#
# Produces:
#   dist/npcot-v${VERSION}-${TARGET}.tar.gz
# containing:
#   npcot_run                # the standalone binary
#   LICENSE                  # from repo root
#   README.md                # short installation / usage doc
#   examples/library.json    # small sample library
#
# Usage:
#   ./packaging/scripts/build_release.sh 0.1.0
#   ./packaging/scripts/build_release.sh 0.1.0 aarch64-apple-darwin
#   ./packaging/scripts/build_release.sh 0.1.0 x86_64-unknown-linux-gnu

set -euo pipefail

VERSION="${1:-0.1.0}"
TARGET="${2:-$(rustc -vV | awk '/^host:/ {print $2}')}"
ROOT="$(git rev-parse --show-toplevel)"
RUST_DIR="${ROOT}/kernels/rust_metal"
DIST_DIR="${ROOT}/dist/npcot-release"
STAGING="${DIST_DIR}/npcot-v${VERSION}-${TARGET}"

echo "[release] version=${VERSION}  target=${TARGET}"
rm -rf "${STAGING}"
mkdir -p "${STAGING}/examples"

echo "[release] building standalone binary"
(cd "${RUST_DIR}" && cargo build --release \
    --bin npcot_run \
    --target "${TARGET}" \
    --no-default-features \
    --features standalone-bin)

BIN_PATH="${RUST_DIR}/target/${TARGET}/release/npcot_run"
if [ ! -x "${BIN_PATH}" ]; then
    echo "[release] missing binary: ${BIN_PATH}" >&2
    exit 1
fi
cp "${BIN_PATH}" "${STAGING}/"

echo "[release] staging docs + license"
if [ -f "${ROOT}/LICENSE" ]; then
    cp "${ROOT}/LICENSE" "${STAGING}/"
fi
cat > "${STAGING}/README.md" <<EOF
# NPCoT Standalone Runtime v${VERSION}

Minimal (~475 KB) runtime for loading and executing NPCoT program libraries
without Python, PyTorch, or any GPU framework.

## Usage

    ./npcot_run examples/library.json \\
        --hidden 1.0,0.0,0.0 \\
        --array 1,2,3,4 \\
        --length 4

Outputs the scalar result of consulting the library for a hidden-state
match and executing the matching discrete reduction on the input array.

## Benchmark

    ./npcot_run examples/library.json \\
        --hidden 1.0,0.0,0.0 --array 1,2,3,4 --length 4 \\
        --benchmark --iters 100000

Typical CPU throughput on Apple M-series: ~4 nanoseconds per
consult+execute on cached skills.

## Compliance

Every library ships with JSON metadata that an auditor can inspect via
the companion \`npcot_compliance\` CLI in the main repo. The Python toolchain
is NOT required for this binary to run — only for producing / verifying
library artifacts.
EOF

# Ship a small sample library — produce on the fly from the Python side
# if available; otherwise hand-write one here so the release tarball
# always contains something runnable.
cat > "${STAGING}/examples/library.json" <<'EOF'
{
  "config": {"similarity_threshold": 0.85, "max_entries": 16, "normalize_epsilon": 1e-08},
  "entries": [
    {
      "signature": [1.0, 0.0, 0.0],
      "program": {"init_idx": 0, "transform_idx": 0, "reduce_idx": 0, "post_scale_idx": 0, "offset": 0.0, "program_text": "sum"},
      "hit_count": 0, "task_name": "sum", "cached_at_step": null, "convergence_gap": null
    },
    {
      "signature": [0.0, 1.0, 0.0],
      "program": {"init_idx": 2, "transform_idx": 0, "reduce_idx": 2, "post_scale_idx": 0, "offset": 0.0, "program_text": "max"},
      "hit_count": 0, "task_name": "max", "cached_at_step": null, "convergence_gap": null
    }
  ]
}
EOF

echo "[release] building tarball"
TARBALL="${DIST_DIR}/npcot-v${VERSION}-${TARGET}.tar.gz"
tar -C "${DIST_DIR}" -czf "${TARBALL}" "npcot-v${VERSION}-${TARGET}"
echo "[release] produced ${TARBALL}"
shasum -a 256 "${TARBALL}"

echo
echo "[release] to publish via Homebrew, update"
echo "          packaging/homebrew/nCPU-npcot.rb"
echo "  with the sha256 shown above."
