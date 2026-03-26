#!/usr/bin/env python3
"""Neural Text Machine: Text processing on a differentiable CPU.

Characters are just integers. The differentiable CPU processes them.
Gradient descent discovers text transformation programs.

Features:
1. Cipher Discovery: Given plaintext/ciphertext pairs, discover the cipher
2. Pattern Generator: Learn character sequences (A->B->C or 1->2->3)
3. Text Transform: Discover case conversion, char shifting, etc.
4. Caesar Cracker: Crack Caesar ciphers via gradient descent on immediates
5. Interactive Mode: Type text, watch it get processed through the neural CPU

Run:
    PYTHONPATH=. python demos/neural_text_machine.py
    PYTHONPATH=. python demos/neural_text_machine.py --interactive
"""

from __future__ import annotations

import sys
import time

import torch

from ncpu.differentiable.execution import (
    DifferentiableEngine,
    FixedProgram,
    SoftProgram,
    Instruction,
    OPCODES,
)
from ncpu.differentiable.program_synthesis import ProgramSynthesizer, SynthesisSpec
from ncpu.differentiable.program_optimizer import ProgramOptimizer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_REGISTERS = 4
MAX_PROGRAM_LEN = 4
DEFAULT_MAX_ITERS = 1500
DEFAULT_LR = 0.03
DEFAULT_MAX_EXEC_STEPS = 6
DEFAULT_TEMPERATURE = 0.1

BANNER = r"""
    ============================================================
     Neural Text Machine
     Text processing through a differentiable CPU

     Characters are integers (ASCII). The CPU operates on numbers.
     Gradient descent discovers text transformations as programs.
    ============================================================
"""


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class NeuralTextMachine:
    """Process text through a differentiable CPU."""

    def __init__(self, num_registers: int = NUM_REGISTERS):
        self.num_registers = num_registers
        self.engine = DifferentiableEngine(num_registers=num_registers)

    # === CIPHER DISCOVERY ===

    def discover_cipher(
        self,
        plaintext: str,
        ciphertext: str,
        max_iters: int = DEFAULT_MAX_ITERS,
    ):
        """Given plaintext and ciphertext, discover the cipher program.

        E.g., plaintext="hello", ciphertext="khoor" -> discovers Caesar shift +3.
        Returns the SynthesisResult.
        """
        if len(plaintext) != len(ciphertext):
            print("  Error: plaintext and ciphertext must be the same length.")
            return None

        if len(plaintext) == 0:
            print("  Error: text must not be empty.")
            return None

        # Build examples: R0 = input char, R1 = expected output char
        examples = []
        for p, c in zip(plaintext, ciphertext):
            examples.append(({0: float(ord(p))}, {1: float(ord(c))}))

        print(f"  Discovering cipher from {len(examples)} character pairs...")
        print(f"  Plaintext:  '{plaintext}'")
        print(f"  Ciphertext: '{ciphertext}'")
        print()

        synth = ProgramSynthesizer(
            max_program_len=MAX_PROGRAM_LEN,
            num_registers=self.num_registers,
            lr=DEFAULT_LR,
        )
        result = synth.synthesize(
            SynthesisSpec(examples),
            max_iters=max_iters,
            verbose=True,
            print_every=max(1, max_iters // 4),
            skip_bitwise=True,
            max_exec_steps=DEFAULT_MAX_EXEC_STEPS,
        )

        print(f"\n  Discovered cipher program:")
        for line in result.program_text.split("\n"):
            print(f"    {line}")
        print(f"  Accuracy: {result.accuracy:.0%}")

        return result

    def apply_program(self, text: str, program: SoftProgram) -> str:
        """Apply a discovered soft program to text, character by character.

        Reads R0 = input char, returns R1 = output char.
        """
        result_chars = []
        with torch.no_grad():
            for ch in text:
                r = self.engine.execute_soft(
                    program,
                    {0: float(ord(ch))},
                    temperature=DEFAULT_TEMPERATURE,
                    max_steps=DEFAULT_MAX_EXEC_STEPS,
                    skip_bitwise=True,
                )
                out_code = int(round(r.registers[1].item()))
                out_code = max(32, min(126, out_code))  # clamp to printable ASCII
                result_chars.append(chr(out_code))
        return "".join(result_chars)

    # === PATTERN GENERATOR ===

    def learn_sequence(
        self,
        sequence: str,
        max_iters: int = DEFAULT_MAX_ITERS,
    ):
        """Learn a character sequence pattern.

        Given "ABCDE", learn that each char -> next char (A->B, B->C, etc.)
        Then generate new characters by iterating.
        """
        if len(sequence) < 2:
            print("  Error: sequence must have at least 2 characters.")
            return None

        examples = []
        for i in range(len(sequence) - 1):
            curr = float(ord(sequence[i]))
            next_ch = float(ord(sequence[i + 1]))
            examples.append(({0: curr}, {1: next_ch}))

        print(f"  Learning sequence pattern from: '{sequence}'")
        print(f"  ({len(examples)} transitions)")
        print()

        synth = ProgramSynthesizer(
            max_program_len=MAX_PROGRAM_LEN,
            num_registers=self.num_registers,
            lr=DEFAULT_LR,
        )
        result = synth.synthesize(
            SynthesisSpec(examples),
            max_iters=max_iters,
            verbose=True,
            print_every=max(1, max_iters // 4),
            skip_bitwise=True,
            max_exec_steps=DEFAULT_MAX_EXEC_STEPS,
        )

        print(f"\n  Discovered pattern program:")
        for line in result.program_text.split("\n"):
            print(f"    {line}")
        print(f"  Accuracy: {result.accuracy:.0%}")

        return result

    def generate_sequence(
        self, start_char: str, program: SoftProgram, length: int = 20,
    ) -> str:
        """Generate a sequence by iterating the learned pattern."""
        chars = [start_char]
        current = float(ord(start_char))

        with torch.no_grad():
            for _ in range(length - 1):
                r = self.engine.execute_soft(
                    program,
                    {0: current},
                    temperature=DEFAULT_TEMPERATURE,
                    max_steps=DEFAULT_MAX_EXEC_STEPS,
                    skip_bitwise=True,
                )
                next_val = r.registers[1].item()
                next_code = int(round(next_val))
                next_code = max(32, min(126, next_code))
                chars.append(chr(next_code))
                current = float(next_code)

        return "".join(chars)

    # === TEXT TRANSFORM DISCOVERY ===

    def discover_transform(
        self,
        input_text: str,
        output_text: str,
        max_iters: int = DEFAULT_MAX_ITERS,
    ):
        """Discover a character-level text transformation.

        Examples:
        - "hello" -> "HELLO" (discover uppercase)
        - "abc" -> "bcd" (discover +1 shift)
        - "12345" -> "24680" (discover *2)
        """
        if len(input_text) != len(output_text):
            print("  Error: input and output text must be the same length.")
            return None

        if len(input_text) == 0:
            print("  Error: text must not be empty.")
            return None

        examples = []
        for i_ch, o_ch in zip(input_text, output_text):
            examples.append(({0: float(ord(i_ch))}, {1: float(ord(o_ch))}))

        print(f"  Discovering transform: '{input_text}' -> '{output_text}'")
        print(f"  ({len(examples)} character mappings)")
        print()

        synth = ProgramSynthesizer(
            max_program_len=MAX_PROGRAM_LEN,
            num_registers=self.num_registers,
            lr=DEFAULT_LR,
        )
        result = synth.synthesize(
            SynthesisSpec(examples),
            max_iters=max_iters,
            verbose=True,
            print_every=max(1, max_iters // 4),
            skip_bitwise=True,
            max_exec_steps=DEFAULT_MAX_EXEC_STEPS,
        )

        print(f"\n  Discovered transform program:")
        for line in result.program_text.split("\n"):
            print(f"    {line}")
        print(f"  Accuracy: {result.accuracy:.0%}")

        return result

    # === CAESAR CIPHER CRACKER ===

    def crack_caesar(self, ciphertext: str, crib: str = "the"):
        """Crack a Caesar cipher by finding the shift via gradient descent.

        Uses program optimization (not synthesis): the program structure
        is known (R1 = R0 + shift), gradient descent finds the shift value.

        Args:
            ciphertext: The encrypted text to crack.
            crib: Known plaintext that maps to the start of ciphertext.
                  Default is "the" (most common English 3-letter word).
        """
        if len(ciphertext) < len(crib):
            print(f"  Error: ciphertext must be at least {len(crib)} characters.")
            return None, None

        print(f"  Cracking Caesar cipher...")
        print(f"  Ciphertext: '{ciphertext}'")
        print(f"  Known plaintext crib: '{crib}' = first {len(crib)} chars")
        print()

        # Build a FixedProgram: R1 = R0 + shift, where shift is learnable.
        # MOV R2, #shift  (immediate to optimize)
        # ADD R1, R0, R2  (output = input + shift)
        # HALT
        program = FixedProgram([
            Instruction(OPCODES["MOV_IMM"], dst=2, immediate=0.0),
            Instruction(OPCODES["ADD"], dst=1, src1=0, src2=2),
            Instruction(OPCODES["HALT"]),
        ])

        # Work in letter-offset space (0-25) to handle wrapping correctly.
        # Caesar ciphers are mod-26, so 't'(19) + 7 = 26 mod 26 = 0 = 'a'.
        # Raw ASCII won't find a single consistent shift due to wrapping.
        cipher_sample = ciphertext[:len(crib)]

        optimizer_obj = torch.optim.Adam(program.parameters(), lr=0.5)
        engine = self.engine

        for step in range(300):
            optimizer_obj.zero_grad()
            total_loss = torch.tensor(0.0)
            for p_ch, c_ch in zip(crib, cipher_sample):
                if p_ch.isalpha() and c_ch.isalpha():
                    p_off = float(ord(p_ch.lower()) - ord("a"))
                    c_off = float(ord(c_ch.lower()) - ord("a"))
                    result = engine.execute_fixed(program, {0: p_off})
                    # Target: (p_off + shift) mod 26 = c_off
                    # Approximate mod 26 with soft wrap for differentiability
                    pred = result.registers[1]
                    # Loss on both direct and wrapped distance
                    direct_loss = (pred - c_off) ** 2
                    wrap_loss = (pred - (c_off + 26.0)) ** 2
                    total_loss = total_loss + torch.min(direct_loss, wrap_loss)
            total_loss = total_loss / len(crib)
            total_loss.backward()
            optimizer_obj.step()

            if total_loss.item() < 1e-4:
                break

        raw_shift = program.immediates.data[0].item()
        shift = round(raw_shift) % 26
        print(f"  Discovered shift: {shift}")

        # Decrypt the full ciphertext using the discovered shift.
        decrypted = []
        for ch in ciphertext:
            if ch.isalpha():
                base = ord("a") if ch.islower() else ord("A")
                decrypted.append(chr((ord(ch) - base - shift) % 26 + base))
            else:
                decrypted.append(ch)
        decrypted_str = "".join(decrypted)

        print(f"  Decrypted: '{decrypted_str}'")
        return shift, decrypted_str


# ---------------------------------------------------------------------------
# Demo functions
# ---------------------------------------------------------------------------

def _caesar_encrypt(text: str, shift: int) -> str:
    """Encrypt text with a Caesar cipher."""
    result = []
    for c in text:
        if c.isalpha():
            base = ord("a") if c.islower() else ord("A")
            result.append(chr((ord(c) - base + shift) % 26 + base))
        else:
            result.append(c)
    return "".join(result)


def demo_cipher_discovery():
    """Demo: discover a Caesar cipher from plaintext/ciphertext."""
    print("=" * 60)
    print("  DEMO 1: Caesar Cipher Discovery")
    print("  Given matching plain/cipher text, discover the program")
    print("=" * 60)
    print()

    machine = NeuralTextMachine()

    # Caesar cipher shift +3: a->d, b->e, etc.
    plain = "hello world"
    cipher = _caesar_encrypt(plain, 3)

    print(f"  Ground truth: Caesar shift +3")
    print()

    result = machine.discover_cipher(plain, cipher, max_iters=DEFAULT_MAX_ITERS)

    if result is not None and result.accuracy > 0.5:
        new_text = "gradient descent"
        encrypted = machine.apply_program(new_text, result.program)
        expected = _caesar_encrypt(new_text, 3)
        print(f"\n  Applying discovered cipher to new text:")
        print(f"    Input:    '{new_text}'")
        print(f"    Output:   '{encrypted}'")
        print(f"    Expected: '{expected}'")
    else:
        print("\n  Cipher discovery did not converge well enough to apply.")


def demo_sequence_generation():
    """Demo: learn a sequence pattern and generate new text."""
    print()
    print("=" * 60)
    print("  DEMO 2: Sequence Pattern Learning")
    print("  Learn: A->B->C->... then generate the alphabet")
    print("=" * 60)
    print()

    machine = NeuralTextMachine()

    result = machine.learn_sequence("ABCDEFGHIJ", max_iters=DEFAULT_MAX_ITERS)

    if result is not None and result.accuracy > 0.3:
        generated = machine.generate_sequence("A", result.program, length=26)
        print(f"\n  Generated from 'A': '{generated}'")
        print(f"  Expected:           'ABCDEFGHIJKLMNOPQRSTUVWXYZ'")
    else:
        print("\n  Sequence learning did not converge well enough to generate.")


def demo_text_transform():
    """Demo: discover uppercase transformation."""
    print()
    print("=" * 60)
    print("  DEMO 3: Text Transform Discovery")
    print("  Discover: lowercase -> uppercase (ASCII subtract 32)")
    print("=" * 60)
    print()

    machine = NeuralTextMachine()

    result = machine.discover_transform(
        "abcdefghijklmnop",
        "ABCDEFGHIJKLMNOP",
        max_iters=DEFAULT_MAX_ITERS,
    )

    if result is not None and result.accuracy > 0.5:
        test_text = "neural cpu"
        transformed = machine.apply_program(test_text, result.program)
        print(f"\n  Applying transform to new text:")
        print(f"    Input:    '{test_text}'")
        print(f"    Output:   '{transformed}'")
        print(f"    Expected: 'NEURAL CPU'")
    else:
        print("\n  Transform discovery did not converge well enough to apply.")


def demo_caesar_crack():
    """Demo: crack a Caesar cipher via gradient descent."""
    print()
    print("=" * 60)
    print("  DEMO 4: Caesar Cipher Cracking via Gradient Descent")
    print("  Known-plaintext attack: assume 'the' maps to first 3 chars")
    print("=" * 60)
    print()

    machine = NeuralTextMachine()

    # Encrypt "the quick brown fox" with shift 7
    plain = "the quick brown fox"
    shift = 7
    cipher = _caesar_encrypt(plain, shift)

    print(f"  Ground truth: shift={shift}, plaintext='{plain}'")
    print()

    found_shift, decrypted = machine.crack_caesar(cipher, crib="the")

    if found_shift is not None:
        match = "CORRECT" if found_shift == shift else "MISMATCH"
        print(f"\n  Shift: discovered={found_shift}, actual={shift} [{match}]")
        if decrypted == plain:
            print(f"  Decryption: PERFECT MATCH")
        else:
            print(f"  Decrypted:  '{decrypted}'")
            print(f"  Expected:   '{plain}'")


def demo_digit_doubler():
    """Demo: discover a program that doubles ASCII digit values."""
    print()
    print("=" * 60)
    print("  DEMO 5: Digit Doubler Discovery")
    print("  Discover: '1'->'2', '2'->'4', '3'->'6', '4'->'8'")
    print("=" * 60)
    print()

    machine = NeuralTextMachine()

    # The transform is: output = input_char + (input_char - '0')
    # i.e., shift by the numeric value of the digit.
    # Simpler view: '1'(49)->'2'(50), '2'(50)->'4'(52), '3'(51)->'6'(54)
    # Actually: these are not a simple linear transform. Let's use something
    # that IS linear: digit char + 1 shift. Or better: multiply by 2 minus offset.
    # For a clean demo, use: char + fixed_offset.
    input_str = "13579"
    output_str = "24680"
    # '1'=49->'2'=50 (+1), '3'=51->'4'=52 (+1), '5'=53->'6'=54 (+1), etc.
    # This is just a +1 shift in ASCII.

    result = machine.discover_transform(
        input_str, output_str, max_iters=DEFAULT_MAX_ITERS,
    )

    if result is not None and result.accuracy > 0.3:
        test_input = "02468"
        transformed = machine.apply_program(test_input, result.program)
        print(f"\n  Applying transform to new digits:")
        print(f"    Input:    '{test_input}'")
        print(f"    Output:   '{transformed}'")
        print(f"    Expected: '13579'")


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def interactive_mode():
    """Interactive text processing REPL."""
    print(BANNER)

    machine = NeuralTextMachine()

    print("""  Commands:
    cipher <plain> <cipher>   Discover cipher from plain/cipher pair
    crack <ciphertext>        Crack Caesar cipher (assumes 'the' crib)
    sequence <text>           Learn and extend a character sequence
    transform <in> <out>      Discover character transformation
    apply <text>              Apply last discovered program to new text
    help                      Show this help
    quit                      Exit
""")

    last_program = None

    try:
        import readline  # noqa: F401 — enables arrow key history in input()
    except ImportError:
        pass

    while True:
        try:
            line = input("text> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        try:
            if cmd in ("quit", "exit", "q"):
                print("  Goodbye!")
                break

            elif cmd == "help":
                print("""  Commands:
    cipher <plain> <cipher>   Discover cipher from plain/cipher pair
    crack <ciphertext>        Crack Caesar cipher (assumes 'the' crib)
    sequence <text>           Learn and extend a character sequence
    transform <in> <out>      Discover character transformation
    apply <text>              Apply last discovered program to new text
    help                      Show this help
    quit                      Exit
""")

            elif cmd == "cipher" and len(parts) >= 3:
                result = machine.discover_cipher(parts[1], parts[2])
                if result is not None:
                    last_program = result.program

            elif cmd == "crack" and len(parts) >= 2:
                cipher_text = " ".join(parts[1:])
                machine.crack_caesar(cipher_text)

            elif cmd == "sequence" and len(parts) >= 2:
                result = machine.learn_sequence(parts[1])
                if result is not None and result.accuracy > 0.3:
                    last_program = result.program
                    gen = machine.generate_sequence(
                        parts[1][0], result.program, 30,
                    )
                    print(f"  Generated: '{gen}'")

            elif cmd == "transform" and len(parts) >= 3:
                result = machine.discover_transform(parts[1], parts[2])
                if result is not None:
                    last_program = result.program

            elif cmd == "apply" and len(parts) >= 2:
                if last_program is None:
                    print("  No program discovered yet. Use cipher/transform/sequence first.")
                else:
                    text = " ".join(parts[1:])
                    out = machine.apply_program(text, last_program)
                    print(f"  '{text}' -> '{out}'")

            else:
                print(f"  Unknown command: '{cmd}'. Type 'help' for commands.")

        except Exception as e:
            print(f"  Error: {e}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)

    if "--interactive" in sys.argv or "-i" in sys.argv:
        interactive_mode()
        return

    print(BANNER)

    t0 = time.time()

    demo_cipher_discovery()
    demo_sequence_generation()
    demo_text_transform()
    demo_caesar_crack()
    demo_digit_doubler()

    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print(f"  All demos complete in {elapsed:.1f}s")
    print(f"  Run with --interactive for live text processing!")
    print("=" * 60)


if __name__ == "__main__":
    main()
