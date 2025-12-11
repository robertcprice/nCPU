"""Train the KVRM-CPU decode LLM.

This script fine-tunes TinyLlama-1.1B with LoRA to decode CPU instructions
into verified registry keys and parameters.

Usage:
    python training/train_decode.py

Requirements:
    - Training data at data/cpu_decode_train.jsonl (50k samples)
    - GPU/MPS with ~4GB VRAM (or CPU with more time)

The model learns to decode assembly instructions like:
    "ADD R3, R1, R2" → {"key": "OP_ADD", "params": {"dest": "R3", "src1": "R1", "src2": "R2"}}
"""

import argparse
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset


# =============================================================================
# Configuration
# =============================================================================

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = Path(__file__).parent.parent / "data" / "cpu_decode_train.jsonl"
OUTPUT_PATH = Path(__file__).parent.parent / "models" / "decode_llm"

# Training hyperparameters (optimized for MPS/Metal)
MAX_SEQ_LENGTH = 128
EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 1
LEARNING_RATE = 3e-4
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01

# LoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


# =============================================================================
# Prompt Formatting
# =============================================================================

def format_prompt(example: dict) -> str:
    """Format a training example as a prompt.

    Uses the same format as inference:
        ### Context:
        {instruction}

        ### Key:
        {json_output}

    Args:
        example: Training example with 'messages' field

    Returns:
        Formatted prompt string
    """
    messages = example["messages"]
    context = ""
    key = ""

    for msg in messages:
        if msg["role"] == "user":
            context = msg["content"]
        elif msg["role"] == "assistant":
            key = msg["content"]

    return f"### Context:\n{context}\n\n### Key:\n{key}"


# =============================================================================
# Training Functions
# =============================================================================

def load_and_prepare_model(device: str):
    """Load base model and apply LoRA.

    Args:
        device: Target device ('cuda', 'mps', or 'cpu')

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading base model: {BASE_MODEL}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map=None
    )
    model = model.to(device)

    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def prepare_dataset(tokenizer, data_path: Path):
    """Load and tokenize the training dataset.

    Args:
        tokenizer: HuggingFace tokenizer
        data_path: Path to JSONL training data

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    print(f"Loading dataset from {data_path}")

    dataset = load_dataset("json", data_files=str(data_path), split="train")
    split = dataset.train_test_split(test_size=0.1, seed=42)

    def tokenize_function(examples):
        texts = [format_prompt({"messages": msgs}) for msgs in examples["messages"]]
        return tokenizer(
            texts,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length"
        )

    print("Tokenizing dataset...")
    tokenized_train = split["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=split["train"].column_names
    )
    tokenized_val = split["test"].map(
        tokenize_function,
        batched=True,
        remove_columns=split["test"].column_names
    )

    print(f"Train samples: {len(tokenized_train)}")
    print(f"Validation samples: {len(tokenized_val)}")

    return tokenized_train, tokenized_val


def train(model, tokenizer, train_dataset, val_dataset, output_path: Path):
    """Run the training loop.

    Args:
        model: The model to train
        tokenizer: Tokenizer for data collation
        train_dataset: Tokenized training data
        val_dataset: Tokenized validation data
        output_path: Where to save the trained model
    """
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        fp16=False,  # MPS doesn't support fp16
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=500,
        seed=42,
        report_to="none",
        dataloader_num_workers=0,
        use_cpu=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("Starting training...")
    start_time = time.time()

    trainer.train()

    elapsed = time.time() - start_time
    print(f"Training complete! Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save the model
    print(f"Saving model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    return trainer


# =============================================================================
# Validation
# =============================================================================

def validate_model(model, tokenizer, device: str):
    """Run quick validation on the trained model.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        device: Device to run on
    """
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    test_instructions = [
        ("MOV R3, 42", "OP_MOV_REG_IMM"),
        ("ADD R0, R1, R2", "OP_ADD"),
        ("CMP R3, R4", "OP_CMP"),
        ("JMP loop", "OP_JMP"),
        ("HALT", "OP_HALT"),
        ("sub r5 r3 r1", "OP_SUB"),  # lowercase
        ("FOO R1, R2", "OP_INVALID"),  # invalid
    ]

    model.eval()
    correct = 0

    for instr, expected_key in test_instructions:
        prompt = f"### Context:\n{instr}\n\n### Key:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the key from the response
        if expected_key in response:
            correct += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"  [{status}] \"{instr}\" → expected {expected_key}")

    accuracy = 100 * correct / len(test_instructions)
    print(f"\nValidation accuracy: {correct}/{len(test_instructions)} ({accuracy:.0f}%)")

    return accuracy


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train KVRM-CPU decode LLM")
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="Path to training data JSONL"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path to save trained model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size per device"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation after training"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("KVRM-CPU DECODE LLM TRAINING")
    print("=" * 60)

    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    print(f"Data path: {args.data}")
    print(f"Output path: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Check data exists
    if not args.data.exists():
        print(f"ERROR: Training data not found at {args.data}")
        print("Run 'python training/generate_cpu_data.py' first to generate training data.")
        return 1

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_and_prepare_model(device)

    # Prepare dataset
    train_dataset, val_dataset = prepare_dataset(tokenizer, args.data)

    # Train
    global EPOCHS, BATCH_SIZE
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    train(model, tokenizer, train_dataset, val_dataset, args.output)

    # Validate
    if not args.skip_validation:
        validate_model(model, tokenizer, device)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved to: {args.output}")
    print("\nTo use the trained model:")
    print("  from kvrm_cpu import KVRMCPU")
    print(f"  cpu = KVRMCPU(mock_mode=False, model_path='{args.output}')")
    print("  cpu.load()")

    return 0


if __name__ == "__main__":
    exit(main())
