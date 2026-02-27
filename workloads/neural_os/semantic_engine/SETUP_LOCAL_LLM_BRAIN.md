# Setting Up the Local LLM Brain for Singularity

## Overview

The Learning Singularity uses a local LLM as its "brain" for intelligent code mutations. This guide shows how to set it up and train it to get smarter over time.

---

## Step 1: Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve
```

## Step 2: Download a Code-Focused Model

```bash
# Option 1: CodeLlama 7B (recommended for speed)
ollama pull codellama:7b

# Option 2: DeepSeek Coder (good at optimization)
ollama pull deepseek-coder:6.7b

# Option 3: CodeLlama 13B (better quality, slower)
ollama pull codellama:13b

# Option 4: Mistral (general purpose, fast)
ollama pull mistral:7b
```

## Step 3: Verify It Works

```bash
# Test the model
ollama run codellama:7b "Write a Python function to check if a number is prime"

# Should output Python code
```

## Step 4: Run the Learning Singularity

```bash
# Quick test (5 minutes)
python3 learning_singularity_20260111.py --duration 5

# Long run (1 hour)
python3 learning_singularity_20260111.py --duration 60

# Daemon mode (runs forever)
python3 learning_singularity_20260111.py --daemon
```

---

## Training the Brain

The system learns from successful optimizations. Here's how it works:

### Experience Collection

Every successful optimization is stored:
```
Original Code → Optimized Code → Speedup → Strategy
```

These are saved to `learning_singularity.db` (SQLite).

### View Collected Experiences

```bash
python3 learning_singularity_20260111.py --stats
```

### Prepare Training Data

After collecting 100+ experiences:

```bash
python3 learning_singularity_20260111.py --train
```

This creates a JSONL file in `training_data/` formatted for fine-tuning.

### Fine-Tune the Model

Create a `Modelfile` for your custom model:

```dockerfile
# Modelfile
FROM codellama:7b

# Set system prompt
SYSTEM """You are a code optimization expert. When given Python code, you
analyze it and produce an optimized version that is faster while maintaining
the same behavior. Focus on algorithm improvements, better data structures,
and Python-specific optimizations like list comprehensions and built-in functions."""

# Adjust parameters for code
PARAMETER temperature 0.3
PARAMETER top_p 0.9
```

Then create your custom model:

```bash
ollama create singularity-brain -f Modelfile
```

### Use Your Trained Model

Update `Config.OLLAMA_MODEL` in `learning_singularity_20260111.py`:

```python
OLLAMA_MODEL = "singularity-brain"
```

---

## Advanced: Full Fine-Tuning

For deeper training, use tools like:

### 1. Axolotl (Recommended)

```bash
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl

# Create config for your data
cat > singularity.yml << EOF
base_model: codellama/CodeLlama-7b-hf
model_type: LlamaForCausalLM
tokenizer_type: CodeLlamaTokenizer

datasets:
  - path: ../training_data/training_*.jsonl
    type: completion

sequence_len: 2048
micro_batch_size: 4
gradient_accumulation_steps: 4
num_epochs: 3
learning_rate: 2e-5
EOF

# Train
python -m axolotl.cli.train singularity.yml
```

### 2. LlamaFactory

```bash
git clone https://github.com/hiyouga/LLaMA-Factory
cd LLaMA-Factory

# Use their training interface
python src/train_bash.py \
    --stage sft \
    --model_name_or_path codellama/CodeLlama-7b-hf \
    --dataset_dir ../training_data \
    --output_dir ../trained_model
```

### 3. Convert to Ollama

After training, convert to Ollama format:

```bash
# Convert to GGUF
python convert.py trained_model --outtype q4_0

# Create Ollama model
ollama create singularity-brain -f Modelfile.trained
```

---

## The Learning Loop

```
┌─────────────────────────────────────────────────────────────┐
│                    LEARNING LOOP                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   1. Run Singularity                                         │
│      └─▶ Optimizes code                                     │
│          └─▶ Records experiences                            │
│                                                              │
│   2. Collect 100+ experiences                               │
│      └─▶ Prepare training data                              │
│                                                              │
│   3. Fine-tune model                                        │
│      └─▶ Model learns optimization patterns                 │
│                                                              │
│   4. Use new model                                          │
│      └─▶ Better mutations                                   │
│          └─▶ More successful optimizations                  │
│              └─▶ Better training data                       │
│                  └─▶ REPEAT                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Complexity Ladder

The system progressively tackles harder problems:

| Level | Description | Examples |
|-------|-------------|----------|
| 1 | Simple functions | Sum, count, filter |
| 2 | Recursive functions | Fibonacci, tree traversal |
| 3 | Multi-function | Stats calculator, parsers |
| 4 | Class-based | Data structures, OOP |
| 5 | Algorithmic | Graphs, sorting, searching |
| 6 | System-level | Multi-file, architecture |

The system advances when it achieves >70% success rate with 10+ improvements.

---

## Auto-Discovery

The system can discover new code to optimize:

```python
# Scan a codebase
explorer = CodeExplorer(llm)
programs = explorer.explore_codebase("/path/to/project")

# Generate new programs with LLM
new_prog = explorer.generate_new_program("recursive_functions")
```

---

## Monitoring

### View Experience Database

```bash
sqlite3 learning_singularity.db

# See total experiences
SELECT COUNT(*) FROM experiences;

# Best optimizations
SELECT original_code, speedup, strategy_used
FROM experiences
ORDER BY speedup DESC
LIMIT 10;

# By strategy
SELECT strategy_used, COUNT(*), AVG(speedup)
FROM experiences
GROUP BY strategy_used;
```

### Watch Live Progress

```bash
# Run with verbose output
python3 learning_singularity_20260111.py --duration 10 2>&1 | tee learning.log
```

---

## Tips

1. **Start with small runs** to build up experiences
2. **Check stats regularly** to see what's working
3. **Fine-tune after 500+ experiences** for best results
4. **Use faster models** (7B) for more iterations
5. **Add your own code** to the exploration path

---

## Files

| File | Purpose |
|------|---------|
| `learning_singularity_20260111.py` | Main system |
| `learning_singularity.db` | Experience database |
| `training_data/*.jsonl` | Training data for fine-tuning |
| `LEARNING_REPORT_*.json` | Run reports |
