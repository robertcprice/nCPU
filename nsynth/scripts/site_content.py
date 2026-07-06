#!/usr/bin/env python3
"""Optional site-content provider for the nsynth site generator.

Reads a copy-writing prompt on stdin, prints ONE short line of copy on stdout.
Wired via  NSYNTH_CONTENT_MODEL="cmd:python3 scripts/site_content.py".

This is the ONLY place a model touches the site pipeline. Structure, sections,
palette, contrast, and links are produced and VERIFIED model-free upstream; this
just fills the one inherently-unverifiable thing — prose — and its output is
sanitized (tags stripped, single line, length-capped) on the Rust side before it
ever reaches the page. If the model is unavailable or errors, we print nothing
and the generator keeps its honest starter scaffold.

Default model: a small LOCAL MLX model (Gemma), per the project's local-first
rule. Override with NSYNTH_CONTENT_MODEL_ID.
"""
import os
import sys

MODEL_ID = os.environ.get("NSYNTH_CONTENT_MODEL_ID", "mlx-community/gemma-2-2b-it-4bit")


def main() -> int:
    prompt = sys.stdin.read().strip()
    if not prompt:
        return 0
    try:
        from mlx_lm import load, generate
    except Exception:
        # mlx_lm not installed -> emit nothing; scaffold survives.
        return 0
    try:
        model, tokenizer = load(MODEL_ID)
        text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=40,
            verbose=False,
        )
    except Exception:
        return 0
    # First non-empty line only; the Rust side sanitizes further.
    for line in text.splitlines():
        line = line.strip().strip('"')
        if line:
            print(line)
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
