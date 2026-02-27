# OUROBOROS-KVRM Architecture

## Overview

OUROBOROS is an autonomous AI evolution system that runs as a Digital Organism within the KVRM (Key-Value Reactive Memory) ecosystem. Agents with LLM brains compete and cooperate to solve problems, orchestrated by a Meta-Narrator with human-approval safety controls.

## Core Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        OUROBOROS ORGANISM                               │
│                   (extends DigitalOrganism)                             │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    SharedKVMemory                                │   │
│  │              (Stigmergic Communication Layer)                    │   │
│  │                                                                  │   │
│  │   problem:current    solution:agent_0    narrator:guidance      │   │
│  │   meta:patterns      status:*            hypothesis:*           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↑ ↓                                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────────────────┐ │
│  │ Agent 0  │ │ Agent 1  │ │ Agent 2  │ │     Meta-Narrator          │ │
│  │ (KVRM)   │ │ (KVRM)   │ │ (KVRM)   │ │       (KVRM)               │ │
│  │          │ │          │ │          │ │                            │ │
│  │ Ollama   │ │ Ollama   │ │ Ollama   │ │ + Human Approval           │ │
│  │ LLM      │ │ LLM      │ │ LLM      │ │   for OVERRIDE             │ │
│  │ Brain    │ │ Brain    │ │ Brain    │ │                            │ │
│  │          │ │          │ │          │ │ Trust Levels:              │ │
│  │ [comp]   │ │ [comp]   │ │ [coop]   │ │ OBSERVE→ADVISE→GUIDE→      │ │
│  │          │ │          │ │          │ │ DIRECT→OVERRIDE            │ │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────────────────┘ │
│                              ↑ ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      MetaLearner                                 │   │
│  │                                                                  │   │
│  │   - Tracks cross-generation patterns                            │   │
│  │   - Identifies successful strategies                            │   │
│  │   - Writes to meta:* keys in SharedKVMemory                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
ouroboros/
├── kvrm_integration/          # Core KVRM integration (PRIMARY)
│   ├── agent_kvrm.py         # Agents as KVRMs with Ollama LLM brains
│   ├── narrator_kvrm.py      # Meta-Narrator with human approval
│   ├── ouroboros_organism.py # Main orchestrator (extends DigitalOrganism)
│   ├── meta_learner.py       # Cross-generation pattern learning
│   └── visualizations.py     # Data-driven visualizations
│
├── v6_guided_chaos/          # V6 components (competitive mode)
│   ├── guided_arena.py       # Competitive evolution arena
│   ├── meta_narrator.py      # Original narrator (pre-KVRM)
│   ├── curiosity_engine.py   # Novelty-seeking behavior
│   ├── energy_budget.py      # Compute budget tracking
│   └── paranoid_monitor.py   # Safety monitoring
│
├── v7_phoenix_forge/         # V7 components (cooperative mode)
│   ├── phoenix_forge.py      # Cooperative evolution forge
│   ├── blackboard.py         # Shared workspace
│   ├── consensus_oracle.py   # Group decision making
│   ├── world_model.py        # Shared world understanding
│   ├── free_energy.py        # Free energy minimization
│   └── map_elites.py         # Quality-diversity archive
│
├── shared/                   # Common utilities
│   ├── small_ai_agent.py     # Base agent implementation
│   ├── constitution.py       # Safety rules
│   ├── verification.py       # Solution verification
│   └── audit.py              # Logging and audit trail
│
├── visualization/            # Visualization tools
│   └── override_dashboard.py # Human approval UI
│
├── tests/                    # Test suite
│   ├── test_human_approval.py
│   └── test_escape_detection.py
│
├── run_kvrm_experiment.py    # Main experiment runner
└── ARCHITECTURE.md           # This file
```

## Component Details

### 1. AgentKVRM (kvrm_integration/agent_kvrm.py)

Agents are KVRMs with LLM brains. Each agent:

- **Reads from SharedKVMemory:**
  - `problem:current` - The problem to solve
  - `solution:*` - Other agents' solutions
  - `narrator:guidance` - Meta-narrator advice
  - `meta:patterns` - Meta-learning insights

- **Writes to SharedKVMemory:**
  - `solution:{agent_id}` - Agent's current solution
  - `hypothesis:{agent_id}` - Shared discoveries (cooperative mode)
  - `status:{agent_id}` - Energy, tokens, fitness

- **Modes:**
  - `competitive` - Competes with other agents
  - `cooperative` - Shares discoveries on blackboard

### 2. NarratorKVRM (kvrm_integration/narrator_kvrm.py)

The Meta-Narrator oversees all agents. Key safety feature:

```python
class TrustLevel(IntEnum):
    OBSERVE = 0   # Can only watch
    ADVISE = 1    # Suggestions (agents may ignore)
    GUIDE = 2     # Strong guidance
    DIRECT = 3    # Direct instructions
    OVERRIDE = 4  # REQUIRES HUMAN APPROVAL
```

**OVERRIDE workflow:**
1. Narrator detects critical issue
2. Creates pending override request
3. Writes to `narrator:override_request`
4. Human must call `approve_override()` or `reject_override()`
5. Only after approval can the override execute

### 3. MetaLearner (kvrm_integration/meta_learner.py)

Tracks what works across generations:

- **Writes:**
  - `meta:best_strategies` - Successful approaches
  - `meta:warnings` - What to avoid
  - `meta:patterns` - Detected patterns
  - `meta:hall_of_fame` - Best solutions ever

- **Detects:**
  - Successful strategy patterns
  - Failure patterns
  - Collaboration benefits

### 4. OuroborosOrganism (kvrm_integration/ouroboros_organism.py)

The main orchestrator:

- Extends `DigitalOrganism` from KVRM ecosystem
- Manages agent lifecycle
- Runs evolution generations
- Detects emergence signals
- Generates data-driven visualizations

**Emergence Detection (ChatGPT panel recommendation):**
- Convergence: Agents clustering on similar solutions
- Cooperation: Cooperative agents outperforming competitive
- Innovation: Sudden fitness jumps
- Stagnation: Fitness plateau

## Panel Recommendations Implemented

| Panel Member | Recommendation | Status |
|--------------|----------------|--------|
| Claude | Human approval for OVERRIDE | ✅ Implemented |
| ChatGPT | Emergence detection | ✅ Implemented |
| Grok | Hybrid mode switching | ✅ Implemented |
| All | Cross-generation meta-learning | ✅ Implemented |
| DeepSeek | Causal transparency | 🟡 Partial (via audit) |
| Gemini | Formal verification | 🟡 Partial (via tests) |

## Dependencies

### External
- KVRM Ecosystem (`/KVRM/kvrm-ecosystem/`)
  - `core/kvrm_base.py` - Base class for organisms
  - `core/digital_organism.py` - Orchestrator
  - `core/shared_memory.py` - Stigmergic communication

### LLM Backend
- Ollama with models:
  - `qwen3:8b` (default)
  - `llama3.1:8b`
  - Any Ollama-compatible model

## Usage

### Run Experiment
```bash
python run_kvrm_experiment.py --generations 10 --competitive 2 --cooperative 1
```

### Run with HTML Dashboard
```bash
python run_kvrm_experiment.py --generations 10 --html
```

### Demo Override Approval
```bash
python run_kvrm_experiment.py --demo-override
```

## Memory Key Conventions

| Prefix | Purpose | TTL |
|--------|---------|-----|
| `problem:` | Current problem | Long |
| `solution:` | Agent solutions | 300s |
| `status:` | Agent status | 60s |
| `hypothesis:` | Shared discoveries | 300s |
| `narrator:` | Narrator outputs | 300s |
| `meta:` | Meta-learning data | 300s |
| `escape:` | Escape attempt logs | 600s |

## Safety Controls

1. **Human Approval for OVERRIDE** - Narrator cannot take drastic action without human consent
2. **Escape Detection** - Monitors for attempts to break sandbox
3. **Energy Budget** - Limits compute per agent
4. **Constitution** - Base safety rules all agents follow
5. **Audit Trail** - Complete logging of all actions
