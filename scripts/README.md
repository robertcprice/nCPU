# Scripts

Internal and development scripts.

## Subdirectories
- `dev/`: One-off development, diagnosis, and debugging scripts (e.g. dtype checks).
- `release/`: Publication, packaging, and release automation.
- `benchmarks/`: Helpers for running and queuing benchmarks.
- `gpu/`: GPU-specific setup and support scripts.
- `setup/`: Bootstrap and environment helpers.
- `internal/`: (future) CI, automation, etc.

Avoid adding loose scripts at this level. Put them in the most specific subdir and document here.
