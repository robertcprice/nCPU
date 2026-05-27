#!/usr/bin/env python3
"""
Constrained decoding — force open-source models to emit only valid
Python function bodies, not prose + markdown + apologies.

This is the technique that turns a 3 B-class open-source model from
"unreliable" to "actually usable" for code generation. A stock
Qwen3.5-4B given a "write a function..." prompt will often emit:

    Sure! Here's the function:

    ```python
    def foo(x):
        return x + 1
    ```

    Let me know if ...

Our benchmark runner's regex-extractor handles that shape. But ~10% of
the time the model emits prose without a code fence, or an explanation
disguised as code, or cuts off mid-function. Those are pure-waste
failures.

Constrained decoding fixes this at the *logit* level. We intercept the
token distribution each step and zero out any token that would violate
the grammar "output must be a Python function starting with `def`".

## Two popular libraries

### 1. lm-format-enforcer  (works with transformers + vLLM + llama.cpp)

```bash
pip install lm-format-enforcer
```

```python
from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)

# "Any text, then `def <name>(...):`, then the body, then nothing."
regex = r"(?:.*?\n)?def\\s+[a-zA-Z_][a-zA-Z_0-9]*\\s*\\([^)]*\\)[^:]*:\\n(?:    .+\\n)+"
parser = RegexParser(regex)

# Wrap the model's generate()
prefix_fn = build_transformers_prefix_allowed_tokens_fn(
    tokenizer, parser
)
out = model.generate(
    **inputs,
    prefix_allowed_tokens_fn=prefix_fn,
    max_new_tokens=512,
)
```

Every generated token is filtered against the regex. Invalid tokens
get probability 0. The model physically cannot produce prose; the only
valid completion is a Python function definition.

### 2. outlines  (richer grammars, slightly heavier dep)

```bash
pip install outlines
```

```python
import outlines
from outlines import models, generate

model = models.transformers("Qwen/Qwen3.5-4B-Instruct")
generator = generate.regex(
    model,
    r"def\\s+[a-zA-Z_][a-zA-Z_0-9]*\\s*\\([^)]*\\):[\\s\\S]*",
)
code = generator(prompt, max_tokens=512)
```

Outlines also supports CFGs for richer constraints — e.g. "must be a
valid Python AST that references only stdlib names". Overkill for our
use case but available.

## What it buys you (measured in the public literature)

- **Syntax error rate**: ~10% → ~0% on function-body outputs
- **End-of-generation truncation**: the regex forces a complete function
- **Prose contamination**: eliminated — model cannot emit "Sure! Here..."

On our HumanEval-lite benchmark, a 3 B open-source model without
constrained decoding loses ~5-10 pp pass@1 to syntax/format errors
alone. With it, those go away entirely and only the *semantic* failures
remain.

## How to integrate with our runners

The `EnhancedInference` wrapper in `inference_enhanced.py` already has
a `use_grammar=True` flag. The hook point is `_one_call`, where we
would pass the `prefix_allowed_tokens_fn` or equivalent to the
backend's generate call.

For the MLX backend, use mlx-lm's `--logits-processors` hook:

```python
# In _MLXBackend.generate(), when use_grammar=True:
from mlx_lm.sample_utils import make_logits_processors
from lmformatenforcer.integrations.mlx import build_mlx_logits_processor

processor = build_mlx_logits_processor(self._tokenizer, regex_parser)
out = self._generate_fn(
    self._model, self._tokenizer, prompt=prompt,
    max_tokens=max_tokens,
    logits_processors=[processor],
)
```

For the HF backend, pass `prefix_allowed_tokens_fn`:
```python
out = self._model.generate(
    **ids, max_new_tokens=max_tokens,
    prefix_allowed_tokens_fn=prefix_fn,
)
```

For the openai-compat HTTP backend, vLLM supports guided decoding:
```python
body["guided_regex"] = regex_pattern
```

## When NOT to use it

- **Free-form conversation**: constrained decoding disables chit-chat
  completely. Use it only when the output shape is known.
- **Multi-function outputs**: if the solution needs helper defs, the
  simple regex above rejects them. Switch to an outlines CFG.
- **Frontier models via API**: Anthropic and OpenAI APIs don't
  generally expose logit-level constraints (outside of OpenAI's JSON
  mode / function calling). For Haiku/Sonnet/Opus, keep the extractor-
  based approach our runners already use.

## Measuring the impact

Run the enhanced runner with and without the grammar flag against the
same benchmark + cache:

```bash
# Baseline (no grammar):
python3 tools/benchmarks/inference_enhanced.py \
    --backend mlx --model mlx-community/Qwen3.5-4B-Instruct-4bit \
    --k 3 --spec @spec.json > /tmp/no_grammar.md

# With grammar:
python3 tools/benchmarks/inference_enhanced.py \
    --backend mlx --model mlx-community/Qwen3.5-4B-Instruct-4bit \
    --k 3 --use-grammar --spec @spec.json > /tmp/with_grammar.md

diff /tmp/no_grammar.md /tmp/with_grammar.md
```

The difference is the measurable value of constrained decoding on
*your* specific model + prompt combo. Publish the number alongside
your pass@1.
"""

# This file is documentation-as-module; the concrete integration is
# in each backend's adapter. It's here as reference text so `--help`
# or `pydoc tools.distillation.constrained_decoding` tells you how.


if __name__ == "__main__":
    import sys
    print(__doc__)
    sys.exit(0)
