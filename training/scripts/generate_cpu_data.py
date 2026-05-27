"""Generate training data for KVRM-CPU decode LLM.

Generates 50,000 instruction samples with augmentation:
- MOV reg, imm: ~10k samples
- MOV reg, reg: ~5k samples
- ADD/SUB/MUL: ~15k samples
- CMP: ~5k samples
- JMP/JZ/JNZ: ~10k samples
- HALT/NOP: ~2k samples
- Invalid: ~3k samples

Augmentations:
- Random capitalization: "ADD", "add", "Add"
- With/without commas: "ADD R3 R1 R2", "ADD R3, R1, R2"
- Extra whitespace variations
- Various label names and numeric addresses
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass


# Register names
REGISTERS = [f"R{i}" for i in range(8)]

# Common label names
LABEL_NAMES = [
    "loop", "start", "end", "done", "exit", "next", "skip", "continue",
    "check", "test", "begin", "finish", "main", "init", "cleanup",
    "process", "handle", "compute", "calculate", "update", "retry"
]

# Variable names for targets
TARGETS = [
    "nums", "data", "items", "values", "arr", "list", "buffer",
    "result", "output", "temp", "sum", "count", "total", "acc"
]


@dataclass
class Sample:
    """Training sample."""
    instruction: str
    key: str
    params: Dict


def random_case(s: str) -> str:
    """Randomly change case of a string."""
    choice = random.choice(["upper", "lower", "title", "mixed"])
    if choice == "upper":
        return s.upper()
    elif choice == "lower":
        return s.lower()
    elif choice == "title":
        return s.title()
    else:
        # Mixed case
        return "".join(c.upper() if random.random() > 0.5 else c.lower() for c in s)


def random_register() -> str:
    """Get a random register."""
    reg = random.choice(REGISTERS)
    return random_case(reg)


def random_immediate() -> int:
    """Generate random immediate value."""
    choice = random.choice(["small", "medium", "large", "negative", "hex_style"])
    if choice == "small":
        return random.randint(0, 10)
    elif choice == "medium":
        return random.randint(0, 255)
    elif choice == "large":
        return random.randint(0, 65535)
    elif choice == "negative":
        return random.randint(-100, -1)
    else:
        return random.randint(0, 255)


def format_immediate(value: int) -> str:
    """Format immediate value (sometimes as hex)."""
    if random.random() < 0.2 and value >= 0:
        return f"0x{value:X}" if random.random() > 0.5 else f"0x{value:x}"
    return str(value)


def random_separator() -> str:
    """Generate random separator between operands."""
    choices = [
        ", ",      # Standard
        ",",       # No space after comma
        " , ",     # Spaces around comma
        "  ,  ",   # Extra spaces
        " ",       # Space only (no comma)
        "  ",      # Double space
    ]
    return random.choice(choices)


def add_whitespace_variation(instr: str) -> str:
    """Add random whitespace variations."""
    if random.random() < 0.2:
        instr = "  " + instr  # Leading spaces
    if random.random() < 0.2:
        instr = instr + "  "  # Trailing spaces
    if random.random() < 0.1:
        instr = "\t" + instr  # Tab prefix
    return instr


def generate_mov_reg_imm() -> Sample:
    """Generate MOV Rd, imm instruction."""
    dest = random_register()
    value = random_immediate()
    value_str = format_immediate(value)

    sep = random_separator()
    op = random_case("MOV")
    instr = f"{op} {dest}{sep}{value_str}"
    instr = add_whitespace_variation(instr)

    return Sample(
        instruction=instr,
        key="OP_MOV_REG_IMM",
        params={"dest": dest.upper(), "value": value}
    )


def generate_mov_reg_reg() -> Sample:
    """Generate MOV Rd, Rs instruction."""
    dest = random_register()
    src = random_register()

    sep = random_separator()
    op = random_case("MOV")
    instr = f"{op} {dest}{sep}{src}"
    instr = add_whitespace_variation(instr)

    return Sample(
        instruction=instr,
        key="OP_MOV_REG_REG",
        params={"dest": dest.upper(), "src": src.upper()}
    )


def generate_arithmetic(op_name: str, key: str) -> Sample:
    """Generate ADD/SUB/MUL instruction."""
    dest = random_register()
    src1 = random_register()
    src2 = random_register()

    sep1 = random_separator()
    sep2 = random_separator()
    op = random_case(op_name)
    instr = f"{op} {dest}{sep1}{src1}{sep2}{src2}"
    instr = add_whitespace_variation(instr)

    return Sample(
        instruction=instr,
        key=key,
        params={"dest": dest.upper(), "src1": src1.upper(), "src2": src2.upper()}
    )


def generate_cmp() -> Sample:
    """Generate CMP instruction."""
    src1 = random_register()
    src2 = random_register()

    sep = random_separator()
    op = random_case("CMP")
    instr = f"{op} {src1}{sep}{src2}"
    instr = add_whitespace_variation(instr)

    return Sample(
        instruction=instr,
        key="OP_CMP",
        params={"src1": src1.upper(), "src2": src2.upper()}
    )


def generate_jump(op_name: str, key: str, labels: Dict[str, int]) -> Sample:
    """Generate JMP/JZ/JNZ instruction."""
    # Choose between label and numeric address
    if labels and random.random() < 0.7:
        # Use a label
        label = random.choice(list(labels.keys()))
        addr = labels[label]
        target_str = random_case(label)
    else:
        # Use numeric address
        addr = random.randint(0, 20)
        target_str = str(addr)

    op = random_case(op_name)
    instr = f"{op} {target_str}"
    instr = add_whitespace_variation(instr)

    return Sample(
        instruction=instr,
        key=key,
        params={"addr": addr}
    )


def generate_halt() -> Sample:
    """Generate HALT instruction."""
    op = random_case("HALT")
    instr = add_whitespace_variation(op)

    return Sample(
        instruction=instr,
        key="OP_HALT",
        params={}
    )


def generate_nop() -> Sample:
    """Generate NOP instruction."""
    op = random_case("NOP")
    instr = add_whitespace_variation(op)

    return Sample(
        instruction=instr,
        key="OP_NOP",
        params={}
    )


def generate_invalid() -> Sample:
    """Generate invalid instruction."""
    invalid_types = [
        # Unknown opcode
        lambda: f"{random.choice(['FOO', 'BAR', 'XYZ', 'MOVE', 'ADDD', 'SUBR'])} R1, R2",
        # Empty
        lambda: "",
        # Just whitespace
        lambda: "   ",
        # Missing operands
        lambda: "MOV R1",
        lambda: "ADD R1, R2",
        lambda: "SUB R1",
        # Invalid register
        lambda: f"MOV R9, 5",
        lambda: f"MOV R10, R1",
        lambda: f"ADD R8, R1, R2",
        # Garbage
        lambda: "$$#@!",
        lambda: "123",
        lambda: random.choice(TARGETS),
        # Partial instruction
        lambda: "MO",
        lambda: "AD",
    ]

    gen = random.choice(invalid_types)
    instr = gen()

    return Sample(
        instruction=instr,
        key="OP_INVALID",
        params={"raw": instr}
    )


def sample_to_jsonl(sample: Sample) -> str:
    """Convert sample to JSONL format for training."""
    messages = [
        {"role": "user", "content": sample.instruction},
        {"role": "assistant", "content": json.dumps({"key": sample.key, "params": sample.params})}
    ]
    return json.dumps({"messages": messages})


def generate_dataset(output_path: Path, total_samples: int = 50000) -> None:
    """Generate the full training dataset."""

    # Target distribution
    distribution = {
        "mov_reg_imm": 10000,
        "mov_reg_reg": 5000,
        "add": 5000,
        "sub": 5000,
        "mul": 5000,
        "cmp": 5000,
        "jmp": 3500,
        "jz": 3500,
        "jnz": 3000,
        "halt": 1000,
        "nop": 1000,
        "invalid": 3000,
    }

    # Create labels dictionary for jump instructions
    labels = {name: random.randint(0, 20) for name in LABEL_NAMES}

    samples = []

    print("Generating samples...")

    # Generate each category
    for _ in range(distribution["mov_reg_imm"]):
        samples.append(generate_mov_reg_imm())

    for _ in range(distribution["mov_reg_reg"]):
        samples.append(generate_mov_reg_reg())

    for _ in range(distribution["add"]):
        samples.append(generate_arithmetic("ADD", "OP_ADD"))

    for _ in range(distribution["sub"]):
        samples.append(generate_arithmetic("SUB", "OP_SUB"))

    for _ in range(distribution["mul"]):
        samples.append(generate_arithmetic("MUL", "OP_MUL"))

    for _ in range(distribution["cmp"]):
        samples.append(generate_cmp())

    # Regenerate labels for each batch to get variety
    for _ in range(distribution["jmp"]):
        labels = {random.choice(LABEL_NAMES): random.randint(0, 20)
                  for _ in range(random.randint(2, 5))}
        samples.append(generate_jump("JMP", "OP_JMP", labels))

    for _ in range(distribution["jz"]):
        labels = {random.choice(LABEL_NAMES): random.randint(0, 20)
                  for _ in range(random.randint(2, 5))}
        samples.append(generate_jump("JZ", "OP_JZ", labels))

    for _ in range(distribution["jnz"]):
        labels = {random.choice(LABEL_NAMES): random.randint(0, 20)
                  for _ in range(random.randint(2, 5))}
        samples.append(generate_jump("JNZ", "OP_JNZ", labels))

    for _ in range(distribution["halt"]):
        samples.append(generate_halt())

    for _ in range(distribution["nop"]):
        samples.append(generate_nop())

    for _ in range(distribution["invalid"]):
        samples.append(generate_invalid())

    # Shuffle samples
    random.shuffle(samples)

    print(f"Generated {len(samples)} samples")

    # Write to JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(sample_to_jsonl(sample) + '\n')

    print(f"Written to {output_path}")

    # Print distribution summary
    print("\nDistribution summary:")
    key_counts = {}
    for sample in samples:
        key_counts[sample.key] = key_counts.get(sample.key, 0) + 1

    for key, count in sorted(key_counts.items()):
        pct = 100 * count / len(samples)
        print(f"  {key}: {count} ({pct:.1f}%)")


def main():
    """Generate training data."""
    output_path = Path(__file__).parent.parent / "data" / "cpu_decode_train.jsonl"

    print("=" * 60)
    print("KVRM-CPU Training Data Generator")
    print("=" * 60)

    generate_dataset(output_path, total_samples=50000)

    print("\nDone!")


if __name__ == "__main__":
    main()
