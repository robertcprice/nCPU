# SPNC Hybrid Review Request: How to Make It Autonomous

## Current State

We have a Self-Programming Neural Computer (SPNC) built on:
- **Layer 1**: 100% accurate KVRM ARM64 execution (8 models)
- **Layer 2**: Program search/synthesis

### Current Capabilities (66.7% accuracy)
- Basic arithmetic: 100% (Double, Triple, Add, Sub, etc.)
- Compound functions: 80% (2x+1, 3x-2, etc.)
- Bitwise: 50% (shifts work, AND/OR with constants fail)

### Critical Failures (12% on novel functions)
- **Cannot discover Square (x*x)** - templates only have x*k, not x*x
- **Cannot discover conditionals** - no if/else templates
- **Cannot discover loops** - no iteration templates
- **Cannot learn from failure** - same failure repeats
- **Cannot generalize** - each function independent
- **Cannot self-improve** - static template set

## The Core Problem

The current SPNC is just a **pattern matcher**, not a true self-programming system.

```
Current: Templates → Match → Done (or fail forever)
Needed:  Observe → Hypothesize → Test → Learn → Improve
```

## What We Want

A system that can:

1. **Discover novel operations** it has never seen
2. **Learn from failures** and improve over time
3. **Generalize** learned patterns to new problems
4. **Self-improve** by adding new capabilities
5. **Actually DO useful things** autonomously

## Specific Questions for Hybrid Review

### 1. How do we enable discovery of novel operations?

Current limitation: Can only find x*k, not x*x (square).

Ideas to evaluate:
- Genetic programming with operation mutation
- Neural program induction
- Symbolic regression
- Program synthesis with SMT solvers

### 2. How do we implement learning from failure?

When synthesis fails, the system should:
- Identify what's missing
- Attempt to create new primitives
- Remember the failure pattern

### 3. How do we enable generalization?

After learning "Double" (x*2) and "Triple" (x*3), the system should:
- Recognize the "multiply by constant" pattern
- Generalize to any "x*k" without explicit training

### 4. How do we enable self-improvement?

The system should:
- Add new templates based on successful discoveries
- Combine known operations in novel ways
- Optimize discovered programs

### 5. What useful tasks can it do autonomously?

Ideas:
- Optimize code snippets for performance
- Discover mathematical identities
- Generate test cases
- Prove program equivalence

## Architecture Options to Evaluate

### Option A: Genetic Programming + Neural Guidance
- Population of random programs
- Neural network predicts fitness
- Evolve toward solution

### Option B: Neural Program Induction
- Train network to output program tokens
- Differentiable execution for gradient-based learning

### Option C: Reinforcement Learning
- State: current program + test cases
- Actions: add/modify/remove operations
- Reward: correctness on test cases

### Option D: Neuro-Symbolic Hybrid
- Neural network generates hypotheses
- Symbolic executor verifies
- Feedback loop for learning

### Option E: Self-Play / Curriculum Learning
- System generates problems for itself
- Gradually increases difficulty
- Learns to solve harder problems

## Success Metrics

1. **Discovery Rate**: % of novel functions solved (target: 80%+)
2. **Learning Speed**: Fewer examples needed after training
3. **Generalization**: Solve unseen functions from same family
4. **Autonomy**: Hours of useful work without human intervention

## Constraints

- Must use KVRM models for execution (100% accurate primitives)
- Must be trainable on consumer GPU (H200 available)
- Must be explainable (show synthesized programs)

## Request

Please provide:

1. **Architectural recommendations** for true autonomy
2. **Training methodology** for self-improvement
3. **Concrete implementation steps** (prioritized)
4. **Failure modes to avoid**
5. **Benchmarks** to measure progress
6. **Revolutionary potential** - what could this become?
