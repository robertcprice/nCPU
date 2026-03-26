# nCPU Demos

This directory contains the fastest way to understand what nCPU can already do.

Only demo source belongs here in git. Generated screenshots, result dumps, local logs, and other ad-hoc artifacts should stay untracked unless they are small, deliberate fixtures. See `docs/REPO_HYGIENE.md` for the general repository policy.

## Start here

If you only run two things, run these:

1. Interactive program discovery

```bash
PYTHONPATH=. python demos/interactive_discovery.py
```

Why it matters:
- you provide examples
- nCPU synthesizes a program live with gradient descent
- you test the discovered program immediately

Good first commands once inside:
- `preset add`
- `preset fib`
- `synthesize`
- `summary`
- `test 13, 21`
- `save exports/fib_session.json`

Example transcript:

```text
ncpu> preset fib
ncpu> synthesize
ncpu> summary
ncpu> test 13, 21
ncpu> export exports/fib_program.asm
```

2. Neural text machine

```bash
PYTHONPATH=. python demos/neural_text_machine.py --interactive
```

Why it matters:
- characters become numeric program inputs
- nCPU discovers ciphers and text transforms from examples
- you can crack Caesar-style shifts, learn sequences, and apply discovered programs to new text

Good first commands once inside:
- `help`
- `cipher hello khoor`
- `summary`
- `apply world`
- `save exports/text_summary.json`
- `sequence ABCDE`
- `transform abc xyz`

Example transcript:

```text
text> cipher hello khoor
text> summary
text> apply world
text> export exports/caesar_program.asm
text> save exports/text_summary.json
```

## Flagship interactive demos

### `interactive_discovery.py`
Program-by-examples REPL for differentiable program synthesis.

```bash
PYTHONPATH=. python demos/interactive_discovery.py
```

### `neural_text_machine.py`
Text processing and text transformation through the differentiable CPU.

```bash
PYTHONPATH=. python demos/neural_text_machine.py
PYTHONPATH=. python demos/neural_text_machine.py --interactive
```

## Systems wow demos

### GPU BusyBox shell

```bash
PYTHONPATH=. python demos/busybox_gpu_demo.py --interactive
```

### Alpine Linux on GPU

```bash
PYTHONPATH=. python demos/alpine_gpu.py --demo
```

### GPU-native tracing / pipeline / self-hosting demos
- `gpu_trace_demo.py`
- `gpu_pipeline_demo.py`
- `self_compile_demo.py`
- `meta_compile_demo.py`

## Research depth demos

### Code in brain / coprocessor demo

```bash
PYTHONPATH=. python demos/demo_code_in_brain.py --help
```

### Program discovery walkthrough

```bash
PYTHONPATH=. python demos/demo_program_discovery.py
```

## Platform matrix

| Area | Best platform | Notes |
|------|---------------|-------|
| Interactive discovery | Cross-platform | Best first-run path for evaluating differentiable synthesis |
| Neural text machine | Cross-platform | Flagship text/program-discovery experience |
| BusyBox / Alpine GPU demos | macOS / Apple Silicon | Metal-backed systems path |
| Coprocessor demo | Cross-platform with model stack | Requires heavier model dependencies and often local weights |

Recommended install paths:

```bash
# Flagship demo surface
pip install -e ".[demo,dev]"

# Broader local environment
pip install -e ".[demo,model,train,dev]"
```

## Recommended newcomer path

1. `interactive_discovery.py`
2. `neural_text_machine.py --interactive`
3. `busybox_gpu_demo.py --interactive`
4. `demo_code_in_brain.py`

If you are evaluating repo novelty, start with 1 and 2.
If you are evaluating systems depth, continue to 3.
If you are evaluating LLM/computation integration, continue to 4.
