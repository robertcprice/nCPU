# Documentation Index

One-screen map of where everything is documented. Paths are repo-relative.

## Plans and direction

| Doc | One line |
|---|---|
| [`ROADMAP.md`](../ROADMAP.md) | The committed execution plan — rungs with definitions of done and verification gates; status per rung. |
| [`docs/native_synthesis_model.md`](native_synthesis_model.md) | Rung 9 design: a program-native coding model — reasoning in program space, JEPA speculation, recursive program networks. |
| [`docs/autoresearch_continuous.md`](autoresearch_continuous.md) | Rung 5 runbook: the nightly attempt→verify→mine→solve→distill loop (produced 85.98% HumanEval on Qwen3.5-4B for $0.39). |

## Component documentation

| Doc | One line |
|---|---|
| [`ncpu/synthesis_api/README.md`](../ncpu/synthesis_api/README.md) | nsynth behind a stdlib-only HTTP API — solves or refuses, transpiles wins, exposes the three memory banks (`/stats`). |
| [`tools/registry/README.md`](../tools/registry/README.md) | Verified-skill registry — trustless contribution (every submission re-executed before storage), `--verify-all` trust sweep, cross-runtime `/library.json`. |
| [`ncpu/mcp_server/README.md`](../ncpu/mcp_server/README.md) | ncpu-synth MCP server — natural language → I/O examples → verified program; honest-refusal contract, never fabricates code. |
| [`tools/pong_synthesis/README.md`](../tools/pong_synthesis/README.md) | Synthesized Pong provenance — 22 rules from example pairs only (14 synthesized + 8 composed), CEGIS loop, reproduction command. |
| [`docs/README.md`](README.md) | Map of the remaining `docs/` subdirectories (architecture, gpu, maintainers, mog, notes, plans, reference). |

## Paper and reproducibility

| Doc | One line |
|---|---|
| [`paper/ncpu_paper.md`](../paper/ncpu_paper.md) | The main paper; its Companion Chapters table indexes every standalone section under `paper/sections/`. |
| [`paper/sections/`](../paper/sections/) | Companion chapters: differentiable programs (§14), Mog synthesis (§15), NPCoT (§15), superblock cache (§16), solver portfolio (§17), sample size (§18), game-scale roadmap (§19), synthesized software (§20). |
| [`paper/sections/section_synthesized_software.md`](../paper/sections/section_synthesized_software.md) | §20: the verified-synthesis stack as a system — formats v1/v2/v3, mined vocabularies, three memory banks, the Pong case study, delivery surfaces. |
| [`REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) | Every numerical claim in `paper/` mapped to its harness command, committed artifact, and regression test. |
| [`paper/build_pdf.sh`](../paper/build_pdf.sh) | Build the publication PDFs (`./paper/build_pdf.sh`; `--all` also builds every companion section). |
