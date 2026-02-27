#!/usr/bin/env python3
"""
Hybrid AI Review for Singularity Core Architecture + Training Scripts
Uses: ChatGPT → Claude → DeepSeek → Grok → Gemini
"""

import os
import json
import time
import traceback
from datetime import datetime
from typing import Optional, Callable

# API clients
import openai
import anthropic
import google.generativeai as genai
import requests

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

AI_STATUS = {"chatgpt": True, "claude": True, "deepseek": True, "grok": True, "gemini": True}


def safe_call(func: Callable, ai_name: str, *args, **kwargs) -> tuple[Optional[str], bool]:
    try:
        result = func(*args, **kwargs)
        return result, True
    except Exception as e:
        print(f"\n⚠️ {ai_name.upper()} FAILED: {str(e)}")
        traceback.print_exc()
        AI_STATUS[ai_name] = False
        return None, False


def call_openai(prompt: str, context: str = "", role_override: str = None) -> str:
    role = role_override or "initial deep analysis"
    print(f"\n{'='*60}")
    print(f"🤖 CHATGPT - {role.upper()}")
    print("="*60)

    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    system_msg = f"""You are an expert in neural networks, program synthesis, and autonomous systems.
Your role: {role}
Provide thorough, deep perspective and analysis. Be comprehensive and detailed.
Focus on: architecture, algorithms, self-improvement mechanisms, and novel approaches."""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"{context}\n\n{prompt}"}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=4000,
        temperature=0.7
    )

    result = response.choices[0].message.content
    print(result)
    return result


def call_claude(prompt: str, context: str = "", previous_review: str = "", role_override: str = None) -> str:
    role = role_override or "best alternative perspective"
    print(f"\n{'='*60}")
    print(f"🧠 CLAUDE - {role.upper()}")
    print("="*60)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    if previous_review:
        full_prompt = f"""
ORIGINAL QUESTION:
{prompt}

PREVIOUS ANALYSIS (ChatGPT):
{previous_review}

YOUR ROLE: {role}

TASKS:
1. Challenge assumptions made in the previous analysis
2. Provide the BEST ALTERNATIVE approach - what's a completely different way to solve this?
3. What would you do DIFFERENTLY and why?
4. Identify weaknesses in the previous approach
5. Propose contrarian ideas that might actually be better
"""
    else:
        full_prompt = f"QUESTION:\n{prompt}\n\nYOUR ROLE: {role}\n\nProvide comprehensive analysis."

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": full_prompt}]
    )

    result = response.content[0].text
    print(result)
    return result


def call_deepseek(prompt: str, previous_reviews: str = "", role_override: str = None) -> str:
    role = role_override or "what both missed - all other options"
    print(f"\n{'='*60}")
    print(f"🔍 DEEPSEEK - {role.upper()}")
    print("="*60)

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    full_prompt = f"""
ORIGINAL QUESTION:
{prompt}

PREVIOUS ANALYSES:
{previous_reviews}

YOUR ROLE: {role}

TASKS:
1. What did BOTH previous reviewers MISS?
2. What are ALL the other options they didn't consider?
3. Provide great depth and nuance to everything
4. Cover edge cases, unconventional approaches, research directions
5. What would a true expert in this field add that's missing?
6. Consider: neuroscience inspiration, mathematical foundations, emergent computation
"""

    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": full_prompt}],
        "max_tokens": 4000,
        "temperature": 0.7
    }

    response = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers=headers,
        json=data,
        timeout=120
    )
    response.raise_for_status()

    result = response.json()["choices"][0]["message"]["content"]
    print(result)
    return result


def call_grok(prompt: str, previous_reviews: str = "", role_override: str = None) -> str:
    role = role_override or "expand everything further"
    print(f"\n{'='*60}")
    print(f"⚡ GROK - {role.upper()}")
    print("="*60)

    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }

    full_prompt = f"""
ORIGINAL QUESTION:
{prompt}

ALL PREVIOUS ANALYSES:
{previous_reviews}

YOUR ROLE: {role}

TASKS:
1. EXPAND on everything discussed so far
2. Add unconventional ideas and edge cases
3. What are the deeper insights being missed?
4. Connect to cutting-edge research and real-world implementations
5. What would make this truly revolutionary vs incremental?
6. Rate each approach discussed (1-10) and explain why
7. What's the moonshot version of this system?
"""

    data = {
        "model": "grok-4-1-fast-reasoning",
        "messages": [{"role": "user", "content": full_prompt}],
        "max_tokens": 4000,
        "temperature": 0.7
    }

    response = requests.post(
        "https://api.x.ai/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=120
    )
    response.raise_for_status()

    result = response.json()["choices"][0]["message"]["content"]
    print(result)
    return result


def call_gemini(prompt: str, all_reviews: str, role_override: str = None) -> str:
    role = role_override or "synthesis + key points + tasks"
    print(f"\n{'='*60}")
    print(f"✨ GEMINI - {role.upper()}")
    print("="*60)

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-pro-preview-06-05")

    full_prompt = f"""
ORIGINAL QUESTION:
{prompt}

ALL PREVIOUS ANALYSES FROM 4 AI MODELS:
{all_reviews}

YOUR ROLE: {role}

FINAL SYNTHESIS TASKS:
1. Find the MOST KEY POINTS from all analyses and EXPAND on them
2. Synthesize everything into coherent, actionable insights
3. Resolve conflicts between different perspectives
4. Create a PRIORITIZED TASK LIST with subtasks

OUTPUT FORMAT:

## KEY INSIGHTS (expanded)
[The most important points, deeply explained]

## SYNTHESIS
[Coherent integration of all perspectives]

## RESOLVED CONFLICTS
[Where reviewers disagreed and the best resolution]

## RECOMMENDED ARCHITECTURE IMPROVEMENTS
[Specific code/architecture changes to make]

## PRIORITIZED TASKS

### Phase 1: Immediate (High Priority)
- [ ] Task 1
  - [ ] Subtask 1.1
  - [ ] Subtask 1.2
- [ ] Task 2

### Phase 2: Core Development
...

### Phase 3: Advanced Features
...

## FINAL VERDICT
[GO/NO-GO with confidence level and key risks]
"""

    response = model.generate_content(full_prompt)
    result = response.text
    print(result)
    return result


def run_hybrid_review(question: str) -> dict:
    print("\n" + "="*70)
    print("🚀 HYBRID AI REVIEW: 5 MODELS SEQUENTIAL ANALYSIS")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}
    accumulated = ""

    # STEP 1: ChatGPT
    print("\n[1/5] ChatGPT - Initial Deep Analysis...")
    chatgpt_result, success = safe_call(call_openai, "chatgpt", question)
    if success:
        results["1_chatgpt"] = chatgpt_result
        accumulated = f"CHATGPT:\n{chatgpt_result}"
    time.sleep(2)

    # STEP 2: Claude
    print("\n[2/5] Claude - Best Alternative...")
    claude_result, success = safe_call(call_claude, "claude", question, "", accumulated)
    if success:
        results["2_claude"] = claude_result
        accumulated += f"\n\nCLAUDE:\n{claude_result}"
    time.sleep(2)

    # STEP 3: DeepSeek
    print("\n[3/5] DeepSeek - What Both Missed...")
    deepseek_result, success = safe_call(call_deepseek, "deepseek", question, accumulated)
    if success:
        results["3_deepseek"] = deepseek_result
        accumulated += f"\n\nDEEPSEEK:\n{deepseek_result}"
    time.sleep(2)

    # STEP 4: Grok
    print("\n[4/5] Grok - Expand Everything...")
    grok_result, success = safe_call(call_grok, "grok", question, accumulated)
    if success:
        results["4_grok"] = grok_result
        accumulated += f"\n\nGROK:\n{grok_result}"
    time.sleep(2)

    # STEP 5: Gemini
    print("\n[5/5] Gemini - Final Synthesis + Tasks...")
    gemini_result, success = safe_call(call_gemini, "gemini", question, accumulated)
    if success:
        results["5_gemini_synthesis"] = gemini_result

    print("\n" + "="*70)
    print("✅ HYBRID REVIEW COMPLETE")
    print("="*70)

    print("\nAI Status:")
    for ai, status in AI_STATUS.items():
        emoji = "✅" if status else "❌"
        print(f"  {emoji} {ai.upper()}")

    return results


# Read the actual code files
def read_file(path):
    try:
        with open(path) as f:
            return f.read()
    except:
        return "[FILE NOT FOUND]"

SINGULARITY_CORE = read_file("singularity_core.py")[:15000]
TRAIN_ADVANCED = read_file("train_advanced.py")[:10000]
TRAIN_H200 = read_file("train_h200.py")[:8000]
MODEL_LOADER = read_file("model_loader.py")[:5000]

ARCHITECTURE_REVIEW_QUESTION = f"""
# Singularity Core Architecture + Training Review

## Current System Status
- **Capability**: 66.67%
- **Layers**: 5/5 active
- **Moonshots**: 6/6 active
- **Trained Models**: MCO 100%, Synthesis 100% accuracy

## Components to Review

### 1. singularity_core.py (Main Architecture)
```python
{SINGULARITY_CORE}
```

### 2. train_advanced.py (Multi-Level Training)
```python
{TRAIN_ADVANCED}
```

### 3. train_h200.py (H200 GPU Training)
```python
{TRAIN_H200}
```

### 4. model_loader.py (Model Integration)
```python
{MODEL_LOADER}
```

## Review Questions

1. **Architecture Improvements**: What structural changes would improve capability from 66.67% to 95%+?

2. **Training Script Quality**: Are the training scripts properly optimized? What's missing?
   - Are we using the right loss functions?
   - Is the data pipeline efficient?
   - Are hyperparameters optimal?

3. **Self-Improvement Loop**: Is the self-improvement loop actually effective? How to make it better?

4. **Novel Algorithm Discovery**: How can we enable discovery of algorithms the system has never seen?

5. **Moonshot Integration**: Are the 6 moonshots (holographic, annealing, omega, evolver, verifier, trained_model) properly integrated?

6. **Critical Missing Pieces**: What critical functionality is missing that would enable true autonomous operation?

7. **Concrete Code Changes**: Provide SPECIFIC code changes to implement the top 5 improvements.

## Target Goals
- Capability: 95%+
- Novel algorithm discovery: 80%+
- True autonomous self-improvement
- No human intervention needed

Please provide:
1. Deep analysis of current architecture
2. Specific weaknesses and how to fix them
3. Prioritized list of improvements
4. Concrete code examples for top improvements
"""


if __name__ == "__main__":
    results = run_hybrid_review(ARCHITECTURE_REVIEW_QUESTION)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    json_path = f"architecture_review_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON saved to: {json_path}")

    # Save as markdown
    md_path = f"ARCHITECTURE_REVIEW_{timestamp}.md"
    with open(md_path, 'w') as f:
        f.write("# Hybrid AI Review: Singularity Core Architecture\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Models: ChatGPT → Claude → DeepSeek → Grok → Gemini\n\n")
        f.write("---\n\n")

        for key, value in results.items():
            title = key.replace("_", " ").upper()
            f.write(f"## {title}\n\n")
            f.write(str(value))
            f.write("\n\n---\n\n")

    print(f"Markdown saved to: {md_path}")
