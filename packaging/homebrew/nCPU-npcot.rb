# Homebrew formula for the NPCoT standalone runtime (N3b).
#
# Installation:
#   brew tap <your-github-org>/nCPU
#   brew install nCPU-npcot
#
# Installs two binaries:
#   npcot_run          — consults a library and executes matching programs
#   (future) npcot_compliance — static-analysis compliance report
#
# After install:
#   npcot_run --help
#   npcot_run path/to/library.json --hidden 1.0,0.0,0.0 --array 1,2,3 --length 3

class NcpuNpcot < Formula
  desc "NPCoT standalone runtime — verified reasoning skills in 475 KB"
  homepage "https://github.com/robertcprice/nCPU"
  license "Apache-2.0"
  version "0.1.0"

  # In a real tap, this `url` would point to a tagged release tarball
  # uploaded to GitHub Releases. For local builds use the editable path
  # via `brew install --build-from-source ./nCPU-npcot.rb`.
  url "https://github.com/robertcprice/nCPU/releases/download/npcot-v0.1.0/npcot-v0.1.0.tar.gz"
  sha256 "REPLACE_WITH_ACTUAL_TARBALL_SHA256_AT_RELEASE_TIME"

  depends_on "rust" => :build

  def install
    # Build the standalone binary with the feature flag that gates off
    # PyO3 / Metal-extension-module dependencies.
    cd "kernels/rust_metal" do
      system "cargo", "build", "--release",
             "--bin", "npcot_run",
             "--no-default-features",
             "--features", "standalone-bin"
      bin.install "target/release/npcot_run"
    end
  end

  test do
    # Smoke test: load a library, consult once.
    lib_json = <<~EOS
      {
        "config": {"similarity_threshold": 0.85, "max_entries": 16, "normalize_epsilon": 1e-08},
        "entries": [
          {
            "signature": [1.0, 0.0, 0.0],
            "program": {"init_idx": 0, "transform_idx": 0, "reduce_idx": 0, "post_scale_idx": 0, "offset": 0.0, "program_text": "sum"},
            "hit_count": 0,
            "task_name": "sum",
            "cached_at_step": null,
            "convergence_gap": null
          }
        ]
      }
    EOS
    (testpath/"lib.json").write(lib_json)
    result = shell_output("#{bin}/npcot_run #{testpath}/lib.json --hidden 1,0,0 --array 1,2,3 --length 3").strip
    assert_equal "6", result
  end
end
