# Benchmarks Directory

This directory contains two kinds of content:

- versioned benchmark scripts and helpers
- local generated benchmark artifacts

Only the scripts and helper code belong in git. Generated result dumps, trajectory captures, remote snapshots, and model outputs are intentionally ignored to keep the repository readable.

If a benchmark run produces a result worth citing, summarize it in `docs/SOME_RESULTS.md` or another tracked report rather than committing the raw artifact tree.
