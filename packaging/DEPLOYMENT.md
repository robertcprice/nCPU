# NPCoT Deployment Options

This project has been verified on **three deployment targets** so far:

* **Local macOS (Apple Silicon + MPS)** — day-to-day dev
* **Linux + NVIDIA CUDA (vast.ai RTX A4000)** — real GPU training / benchmarks
* **Browser (WASM)** — 77 KB client-side library runtime

vast.ai is ONE of many ways to run the GPU-dependent pieces. Here is the
full matrix of where NPCoT runs today and what each option costs.

## What needs what

| Operation                                    | GPU? | Typical cost |
|----------------------------------------------|------|--------------|
| Library fast-path consult (cached skills)    | No   | CPU ~4 ns    |
| Native Rust standalone runtime               | No   | 475 KB binary |
| Browser-side library consult                 | No   | 77 KB WASM   |
| `ArrayExecutableThoughtHead` soft forward    | Any  | ~0.5 ms/call on CPU, ~15 ms/call on GPU |
| Training a new library on curriculum         | Any  | seconds–minutes on CPU, sub-second on GPU |
| Wrapping a real LLM + HumanEval / MBPP eval  | Yes  | $5–$15 for a full Qwen3.5-1.5B run |
| Multi-GPU coprocessor training sweep         | Yes  | $20–$100 for A100 multi-epoch |

**Short version**: inference on a trained library is free. Only the
*training* side of NPCoT wants a GPU.

## Option 1 — vast.ai (cheapest, most control)

Already demonstrated on April 18, 2026 — see `artifacts/vast_ai_run/`.

```bash
export VASTAI=/Users/bobbyprice/Library/Python/3.14/bin/vastai

./packaging/scripts/vast_run.sh tests
./packaging/scripts/vast_run.sh bench3
./packaging/scripts/vast_run.sh humaneval Qwen/Qwen3.5-1.5B my_library.json
./packaging/scripts/vast_run.sh mbpp Qwen/Qwen3.5-1.5B my_library.json
```

* **Price**: $0.08–$0.15/hr for a 16 GB GPU, $0.30–$1.00/hr for A100 40/80 GB
* **Verified cost for our test suite run**: $0.16 total
* **Pros**: cheapest, SSH access, easy to iterate interactively
* **Cons**: need to manage instance lifecycle (the script handles this)

## Option 2 — Modal (zero ops, pay per second)

```bash
pip install modal
modal setup  # first time only

modal run packaging/modal_run.py::run_tests
modal run packaging/modal_run.py::run_bench3
modal run packaging/modal_run.py::run_humaneval --model Qwen/Qwen3.5-1.5B --library /path/to/lib.json
```

* **Price**: ~$0.60/hr for A10G, ~$3.75/hr for A100 (on-demand)
* **Pros**: zero instance management, GPU spins up on function call
* **Cons**: 2–5× more expensive than vast.ai for sustained usage

## Option 3 — RunPod (spot pricing, GPU-by-minute)

```bash
# Provision any GPU pod from the UI (recommend: RTX 3090 community cloud).
# SSH in. Then:

git clone <your-mirror>/nCPU
cd nCPU
pip install torch transformers datasets pytest
pytest tests/self_optimizing/ -q
python3 -m benchmarks.benchmark_npcot_coding_bench --n-problems 200
python3 -m ncpu.self_optimizing.humaneval_runner --no-library --max-problems 100
```

* **Price**: ~$0.20/hr for RTX 3090 community cloud, ~$1.60/hr for A100
* **Pros**: very low spot prices, enterprise support path
* **Cons**: fewer regions than vast.ai, more UI-driven

## Option 4 — Local Apple Silicon (development)

```bash
python3 -m pytest tests/self_optimizing/ -q              # 1.5 s for 200 tests
python3 -m demos.npcot_scale_practicality                # 1000-problem smoke
python3 -m benchmarks.benchmark_npcot_library --device mps --iters 200
```

* **Price**: free
* **Pros**: immediate iteration, no cloud auth
* **Cons**: no CUDA tests, can't run full-size LLMs fast enough for batch eval
* **Good for**: library runtime, soft-forward profiling, test suite development

## Option 5 — Serverless / Lambda / Cloud Run (library-only inference)

The 475 KB standalone Rust binary and 77 KB WASM binary are both fast
enough to serve library consults from any serverless platform:

* AWS Lambda: ship the Rust binary as a custom runtime. ~1 ms cold start.
* Cloudflare Workers: ship the WASM binary. ~0.3 ms per request.
* Cloud Run: the stdlib Python HTTP server in a 50 MB image.

No GPU needed because the library *itself* is a lookup table — inference
is table lookup + a short arithmetic loop.

## Option 6 — Browser / client-side (WASM)

```javascript
import init, { NpcotRuntime } from './npcot_wasm.js';

await init();
const library = await fetch('/library.json').then(r => r.text());
const runtime = new NpcotRuntime(library);
const result = runtime.consult(
    new Float32Array([1.0, 0.0, 0.0]),
    new Float32Array([1.0, 2.0, 3.0]),
    3
);
```

* **Price**: free (client-side)
* **Binary**: 77 KB WASM
* **Pros**: private-by-default inference, no round-trip to server
* **Cons**: library size limited by download time

## Which to pick

| Scenario                                       | Pick            |
|------------------------------------------------|-----------------|
| Day-to-day dev on a Mac                        | Option 4 local  |
| One-off benchmark run with tight budget        | Option 1 vast.ai|
| CI / automated GPU validation                  | Option 2 Modal  |
| Long training sweep                            | Option 3 RunPod |
| Production library-inference API               | Option 5 serverless |
| Ship to end users' browsers                    | Option 6 WASM   |

## Reproducibility

Every script / doc above points to the same library artifact format
(`~/.nCPU_program_library.json`). A library produced by Option 1 can be
consumed unchanged by Options 4, 5, 6. Library fingerprints
(`library.fingerprint()`) give a stable ID you can pin across deployments.
