import os
#!/usr/bin/env python3
"""Quick hybrid review - works without google.generativeai"""
import openai
import anthropic
import requests
import json

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")

# Read the training script
import sys
script_to_review = sys.argv[1] if len(sys.argv) > 1 else 'train_h200.py'
with open(script_to_review) as f:
    code = f.read()
print(f"Reviewing: {script_to_review}")

REVIEW_DOC = f"""
# SEMANTIC SYNTHESIZER H200 TRAINING SCRIPT REVIEW

## PURPOSE OF THIS SYSTEM

This is the **Singularity Core** - an autonomous semantic program synthesis system.
It implements Grok's 5-layer architecture for recursive self-improvement:

### Layer Stack:
- **Layer 4: Epistemic Frontier** - Discovers unknown unknowns via cross-domain bisociation
- **Layer 3: Meta-Cognitive Orchestrator** - Neural RL that learns synthesis strategies
- **Layer 2: Compositional Discovery Engine** - Algebraic rewrites + MDL compression
- **Layer 1: Semantic Operation Network** - Operations as mathematical objects
- **Layer 0: KVRM** - 100% accurate execution substrate

### Moonshot Accelerators:
- **Holographic Programs** - O(1) program search via superposition
- **Thermodynamic Annealing** - Phase transitions discover loops/conditionals
- **Omega Machine** - Self-modifying code (rewrites its own architecture)
- **EvoRL** - Genetic evolution of RL policies

### WHY THIS IS THE SINGULARITY:
1. System improves itself (Omega Machine)
2. Each improvement makes it BETTER at improving (recursive)
3. Holographic/Annealing find things gradient descent cannot
4. No ceiling on capability growth
5. Exponential acceleration

## THE TRAINING SCRIPT TO REVIEW

```python
{code}
```

## CRITICAL REVIEW POINTS:
1. Will this run correctly on H200 with CUDA?
2. Any bugs that would crash training?
3. Is the training loop correct?
4. Can it achieve 100% accuracy?
5. Memory issues with 143GB VRAM?
6. Dataset generation correct?
7. Hyperparameters reasonable?
8. Checkpointing correct?

## REQUIRED OUTPUT:
1. Critical bugs found
2. Improvements needed
3. GO/NO-GO verdict for H200 deployment
"""

print("="*70)
print("HYBRID AI REVIEW: Semantic Synthesizer Training Script")
print("="*70)

# 1. ChatGPT Review
print("\n[1/3] 🤖 CHATGPT REVIEW...")
client = openai.OpenAI(api_key=OPENAI_API_KEY)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are an expert ML engineer. Review this training script for bugs and issues. Be concise."},
        {"role": "user", "content": REVIEW_DOC}
    ],
    max_tokens=2000
)
chatgpt_review = response.choices[0].message.content
print(chatgpt_review)

# 2. Claude Review
print("\n[2/3] 🧠 CLAUDE REVIEW...")
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=2000,
    messages=[{"role": "user", "content": f"""
{REVIEW_DOC}

PREVIOUS REVIEW (ChatGPT):
{chatgpt_review}

Your task: Add your perspective, address ChatGPT's points, find issues they missed. Give GO/NO-GO verdict.
"""}]
)
claude_review = response.content[0].text
print(claude_review)

# 3. Grok Final Verdict
print("\n[3/3] ⚡ GROK FINAL VERDICT...")
headers = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
data = {
    "model": "grok-3-latest",
    "messages": [{"role": "user", "content": f"""
{REVIEW_DOC}

CHATGPT REVIEW:
{chatgpt_review}

CLAUDE REVIEW:
{claude_review}

YOUR TASK: Synthesize all reviews. Give FINAL GO/NO-GO verdict with confidence level.
List the TOP 3 things to fix before deployment.
"""}],
    "max_tokens": 2000
}
response = requests.post("https://api.x.ai/v1/chat/completions", headers=headers, json=data, timeout=120)
grok_review = response.json()["choices"][0]["message"]["content"]
print(grok_review)

print("\n" + "="*70)
print("REVIEW COMPLETE")
print("="*70)
