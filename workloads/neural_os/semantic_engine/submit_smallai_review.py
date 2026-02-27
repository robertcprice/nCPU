#!/usr/bin/env python3
"""Submit OUROBOROS Small AI Agents plan to 6-AI Hybrid Review Panel."""

import sys
sys.path.insert(0, '/Users/bobbyprice/projects/KVRM/kvrm-llm-compiler/staged_classifier')

from hybrid_review_spnc import run_hybrid_review
from datetime import datetime

PLAN_PATH = "/Users/bobbyprice/.claude/plans/hidden-giggling-flame.md"

with open(PLAN_PATH, 'r') as f:
    plan_content = f.read()

REVIEW_PROMPT = f"""
# OUROBOROS V6/V7 with SMALL AI AGENTS - Critical Architecture Review

## MAJOR CHANGE: Agents Are Now Small AIs

We've fundamentally changed the agent architecture:

**BEFORE:** Agents = passive code snippets, external LLM proposes mutations
**AFTER:** Each agent = small AI model (TinyLlama/Phi-2) that can REASON, LEARN, PLAN

Each agent now has:
- Its own brain (small LLM like TinyLlama 1.1B or Phi-2 2.7B)
- World model (beliefs about code)
- Memory (past experiences)
- Strategy network (how to approach problems)
- TWO-LEVEL energy budget (mutation cost + program compute cost)

## Review Questions

1. **Is the Small AI agent architecture viable?**
   - Can 100 agents each with TinyLlama brains run feasibly?
   - What's the compute/memory requirement?
   - Should we use smaller models? LoRA adapters? Shared weights?

2. **How should agent brains evolve?**
   - Evolve the prompts/strategies?
   - Fine-tune the small models?
   - Evolve LoRA adapters?
   - Neuroevolution of weights?

3. **Two-level energy budget - is this sound?**
   - Level 1: Cost to think/mutate (agent's metabolism)
   - Level 2: Cost of programs it creates (efficient solutions favored)
   - Creates pressure for both smart agents AND efficient code

4. **V6 vs V7 comparison with AI agents**
   - V6: 100-200 competing small AIs (fight + curiosity)
   - V7: 50-100 cooperating small AIs (active inference + blackboard)
   - Will competition or cooperation produce better results?

5. **Safety with AI agents vs code agents**
   - AI agents can REASON - does this increase escape risk?
   - Should agent brains be frozen or learnable?
   - How do we prevent agents from gaming their own evaluation?

6. **What are we missing?**
   - Attack vectors we haven't considered?
   - Better small model choices?
   - Alternative agent architectures?

## The Full Plan

{plan_content}

---

Provide:
- Specific technical recommendations for small AI agents
- Model selection guidance (TinyLlama vs Phi-2 vs others)
- How to evolve agent brains safely
- Overall rating (1-10) and GO/NO-GO verdict
"""

if __name__ == "__main__":
    print("=" * 70)
    print("SUBMITTING SMALL AI AGENTS PLAN TO 6-AI HYBRID REVIEW PANEL")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print()
    print("This will query: ChatGPT → Claude → DeepSeek → Grok → Gemini")
    print("Estimated time: 5-10 minutes")
    print("=" * 70)
    print()
    
    result = run_hybrid_review(REVIEW_PROMPT)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"SMALLAI_REVIEW_{timestamp}.md"
    
    with open(output_file, 'w') as f:
        f.write(f"# OUROBOROS Small AI Agents - 6-AI Hybrid Panel Review\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        f.write("---\n\n")
        if isinstance(result, dict):
            for key, value in result.items():
                f.write(f"## {key}\n\n{value}\n\n")
        else:
            f.write(str(result))
    
    print(f"\n\nResults saved to: {output_file}")
