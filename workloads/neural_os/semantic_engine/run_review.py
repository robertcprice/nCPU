#!/usr/bin/env python3
"""Run hybrid review on train_h200.py"""
import sys
sys.path.insert(0, '/Users/bobbyprice/projects/KVRM/kvrm-llm-compiler/staged_classifier')
from hybrid_review import run_hybrid_review

# Read the training script
with open('/Users/bobbyprice/projects/KVRM/kvrm-spnc/semantic_engine/train_h200.py') as f:
    code = f.read()

review_input = f"""
# H200 Training Script Review Request

## File: train_h200.py
## Purpose: Train semantic synthesizer neural networks to 100% accuracy on H200 GPU

## Code to Review:

```python
{code}
```

## Critical Review Points:
1. Will this script run correctly on H200 with CUDA?
2. Are there any bugs that would cause training to fail?
3. Is the training loop correct?
4. Will it actually achieve 100% accuracy?
5. Are there memory issues with the H200's 143GB VRAM?
6. Is the dataset generation correct?
7. Are the hyperparameters reasonable?
8. Will checkpointing work correctly?
9. Any edge cases that could crash training?

## Required: GO/NO-GO verdict for deploying to H200
"""

results = run_hybrid_review(review_input)
print("\n\n" + "="*70)
print("REVIEW COMPLETE - Check above for GO/NO-GO verdict")
print("="*70)
