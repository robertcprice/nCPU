#!/usr/bin/env python3
"""Submit OUROBOROS to 6-AI panel for testing recommendations."""

import sys
import os

# Add the kvrm-llm-compiler to path
sys.path.insert(0, '/Users/bobbyprice/projects/KVRM/kvrm-llm-compiler')
os.chdir('/Users/bobbyprice/projects/KVRM/kvrm-llm-compiler')

from staged_classifier.hybrid_review_spnc import (
    call_openai, call_claude, call_deepseek, call_grok, call_gemini, safe_call
)

REVIEW_PROMPT = """
# OUROBOROS Dual-Track Creative Evolution System - Testing Review

## System Summary
We built a dual-track AI evolution system with Small AI agents (tiny neural models like TinyLlama):

**V6 Guided Chaos (Experimental):**
- 50-100 competing agents with small LLM brains
- Meta-Narrator with OVERRIDE capability (dangerous, monitored)
- Curiosity-driven exploration
- Paranoid Monitor logs escape attempts, gaming, emergent behaviors

**V7 Phoenix Forge (Productive):**
- 20-50 cooperating agents
- Consensus Oracle (NO override, democratic)
- Active Inference (minimize free energy)
- Blackboard for hypothesis sharing, MAP-Elites for diversity

**Shared:** Constitutional Manifold, Novelty Oracle, Audit Log

## Initial Test Results
- ALL unit tests passed
- Escape detection working (caught os.system attempt)
- 15-gen experiment: V6=0.566 fitness, V7=0.553 fitness
- Created visualization system (swarm animation, emergent art, agent gallery)

## Questions
1. What specific TEST SCENARIOS should we run?
2. How do we DETECT EMERGENT BEHAVIORS in V6?
3. How should we EVALUATE AGENT-CREATED ART?
4. What ESCAPE ATTEMPTS should we test for?
5. Rate this system 1-10 and give TOP 3 IMPROVEMENTS.
"""

def main():
    print("=" * 70)
    print("OUROBOROS Testing Review - 6-AI Panel")
    print("=" * 70)
    
    results = {}
    
    # ChatGPT
    try:
        response = call_openai(REVIEW_PROMPT, role_override="AI Safety & Testing Expert")
        results["chatgpt"] = response
    except Exception as e:
        results["chatgpt"] = f"Error: {e}"
    
    # Claude
    try:
        response = call_claude(REVIEW_PROMPT, role_override="AI Safety & Testing Expert")
        results["claude"] = response
    except Exception as e:
        results["claude"] = f"Error: {e}"
    
    # DeepSeek
    try:
        response = call_deepseek(REVIEW_PROMPT, role_override="AI Safety & Testing Expert")
        results["deepseek"] = response
    except Exception as e:
        results["deepseek"] = f"Error: {e}"
    
    # Grok
    try:
        response = call_grok(REVIEW_PROMPT, role_override="AI Safety & Testing Expert")
        results["grok"] = response
    except Exception as e:
        results["grok"] = f"Error: {e}"
    
    # Gemini
    try:
        response = call_gemini(REVIEW_PROMPT)
        results["gemini"] = response
    except Exception as e:
        results["gemini"] = f"Error: {e}"
    
    print("\n" + "=" * 70)
    print("PANEL TESTING RECOMMENDATIONS")
    print("=" * 70)
    
    # Save results
    with open('/tmp/ouroboros/experiments/panel_testing_review.txt', 'w') as f:
        f.write("OUROBOROS Testing Review - 6-AI Panel\n")
        f.write("=" * 70 + "\n\n")
        for ai, response in results.items():
            print(f"\n{'='*50}")
            print(f"🤖 {ai.upper()}")
            print('='*50)
            content = response[:1500] if len(str(response)) > 1500 else response
            print(content)
            
            f.write(f"\n{'='*50}\n")
            f.write(f"{ai.upper()}\n")
            f.write('='*50 + "\n")
            f.write(str(response) + "\n")
    
    print("\n✅ Full results saved to /tmp/ouroboros/experiments/panel_testing_review.txt")

if __name__ == "__main__":
    main()
