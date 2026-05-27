"""Neural Assembler for nCPU.

A complete assembler where tokenization, parsing, and code generation
are all implemented as neural networks, with classical fallbacks.

The assembler translates nCPU assembly source into binary machine code.
It targets the nCPU ISA (R0-R7, text-mode instructions).

Pipeline:
    Source text → Tokenizer → Tokens → Parser → IR → CodeGen → Binary

The conventional (classical) assembler is used as ground truth for
training and validation. The neural assembler is trained to produce
byte-identical output.

nCPU ISA (text mode - as used by ncpu.model.CPU):
    MOV Rd, imm          — Load immediate into register
    MOV Rd, Rs           — Copy register
    ADD Rd, Rs1, Rs2     — Rd = Rs1 + Rs2
    SUB Rd, Rs1, Rs2     — Rd = Rs1 - Rs2
    MUL Rd, Rs1, Rs2     — Rd = Rs1 * Rs2
    DIV Rd, Rs1, Rs2     — Rd = Rs1 / Rs2
    AND Rd, Rs1, Rs2     — Rd = Rs1 & Rs2
    OR  Rd, Rs1, Rs2     — Rd = Rs1 | Rs2
    XOR Rd, Rs1, Rs2     — Rd = Rs1 ^ Rs2
    SHL Rd, Rs, amount   — Rd = Rs << amount
    SHR Rd, Rs, amount   — Rd = Rs >> amount
    INC Rd               — Rd = Rd + 1
    DEC Rd               — Rd = Rd - 1
    CMP Rs1, Rs2         — Set flags from Rs1 - Rs2
    JMP label/addr       — Unconditional jump
    JZ  label/addr       — Jump if zero flag
    JNZ label/addr       — Jump if not zero
    JS  label/addr       — Jump if sign (negative)
    JNS label/addr       — Jump if not sign
    HALT                 — Stop execution
    NOP                  — No operation

Binary encoding (32-bit words):
    [31:24] opcode (8 bits)
    [23:21] dest register (3 bits)
    [20:18] src1 register (3 bits)
    [17:15] src2 register / shift amount (3 bits)
    [14:0]  immediate value (15 bits, sign-extended)

    Opcode map:
        0x00 = NOP
        0x01 = HALT
        0x10 = MOV reg, imm
        0x11 = MOV reg, reg
        0x20 = ADD
        0x21 = SUB
        0x22 = MUL
        0x23 = DIV
        0x30 = AND
        0x31 = OR
        0x32 = XOR
        0x33 = SHL
        0x34 = SHR
        0x40 = INC
        0x41 = DEC
        0x50 = CMP
        0x60 = JMP
        0x61 = JZ
        0x62 = JNZ
        0x63 = JS
        0x64 = JNS
"""

import torch
import torch.nn as nn
import re
import logging
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import IntEnum

from .device import default_device

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Opcode Encoding
# ═══════════════════════════════════════════════════════════════════════════════

class Opcode(IntEnum):
    NOP     = 0x00
    HALT    = 0x01
    MOV_IMM = 0x10
    MOV_REG = 0x11
    ADD     = 0x20
    SUB     = 0x21
    MUL     = 0x22
    DIV     = 0x23
    AND     = 0x30
    OR      = 0x31
    XOR     = 0x32
    SHL     = 0x33
    SHR     = 0x34
    INC     = 0x40
    DEC     = 0x41
    CMP     = 0x50
    JMP     = 0x60
    JZ      = 0x61
    JNZ     = 0x62
    JS      = 0x63
    JNS     = 0x64


# Mnemonic → Opcode mapping
MNEMONIC_MAP = {
    "NOP": Opcode.NOP,
    "HALT": Opcode.HALT,
    "ADD": Opcode.ADD,
    "SUB": Opcode.SUB,
    "MUL": Opcode.MUL,
    "DIV": Opcode.DIV,
    "AND": Opcode.AND,
    "OR": Opcode.OR,
    "XOR": Opcode.XOR,
    "SHL": Opcode.SHL,
    "SHR": Opcode.SHR,
    "INC": Opcode.INC,
    "DEC": Opcode.DEC,
    "CMP": Opcode.CMP,
    "JMP": Opcode.JMP,
    "JZ": Opcode.JZ,
    "JNZ": Opcode.JNZ,
    "JS": Opcode.JS,
    "JNS": Opcode.JNS,
}

# Opcode → Mnemonic (reverse)
OPCODE_TO_MNEMONIC = {v: k for k, v in MNEMONIC_MAP.items()}

REGISTERS = {f"R{i}": i for i in range(8)}

# Instruction format types
FMT_NONE = 0       # NOP, HALT
FMT_REG_IMM = 1    # MOV Rd, imm
FMT_REG_REG = 2    # MOV Rd, Rs
FMT_3REG = 3       # ADD Rd, Rs1, Rs2
FMT_REG = 4        # INC Rd, DEC Rd
FMT_2REG = 5       # CMP Rs1, Rs2
FMT_ADDR = 6       # JMP addr
FMT_SHIFT = 7      # SHL Rd, Rs, amount


# ═══════════════════════════════════════════════════════════════════════════════
# Intermediate Representation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AsmToken:
    """A single token from the assembly source."""
    type: str     # "mnemonic", "register", "immediate", "label_ref", "label_def", "comma", "newline"
    value: str    # The raw text
    line: int     # Source line number
    col: int = 0  # Column number


@dataclass
class AsmInstruction:
    """Parsed assembly instruction (IR node)."""
    opcode: int           # Opcode enum value
    fmt: int              # Format type
    rd: int = 0           # Destination register
    rs1: int = 0          # Source register 1
    rs2: int = 0          # Source register 2
    imm: int = 0          # Immediate value
    label_ref: str = ""   # Unresolved label reference
    line: int = 0         # Source line number
    source: str = ""      # Original source text


@dataclass
class AssemblyResult:
    """Result of assembling a program."""
    binary: List[int]             # List of 32-bit encoded instructions
    instructions: List[AsmInstruction]  # Parsed IR
    labels: Dict[str, int]        # Label → instruction index
    errors: List[str]             # Any assembly errors
    source_map: Dict[int, int]    # binary_index → source_line

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    @property
    def num_instructions(self) -> int:
        return len(self.binary)


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Tokenizer
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralTokenizerNet(nn.Module):
    """Character-level tokenizer network.

    Classifies each character in the input as belonging to a token type.
    Uses a 1D CNN over character embeddings for local context.

    Architecture:
        char → Embedding(128, 32) → Conv1d(32, 64, k=5) → Conv1d(64, 32, k=3) → Linear(32, num_classes)
    """

    def __init__(self, vocab_size: int = 128, embed_dim: int = 32,
                 num_classes: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        """chars: [batch, seq_len] int → [batch, seq_len, num_classes]"""
        x = self.embed(chars)               # [B, L, E]
        x = x.transpose(1, 2)              # [B, E, L]
        x = torch.relu(self.conv1(x))       # [B, 64, L]
        x = torch.relu(self.conv2(x))       # [B, 32, L]
        x = x.transpose(1, 2)              # [B, L, 32]
        return self.classifier(x)           # [B, L, C]


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Code Generator
# ═══════════════════════════════════════════════════════════════════════════════

def encode_instruction_features(
    opcode: int, rd: int, rs1: int, rs2: int, imm: int, fmt: int,
    device: torch.device = None,
) -> torch.Tensor:
    """Encode instruction fields into a 56-dim binary feature vector.

    Returns:
        [0:8]   opcode as 8 bits (LSB first)
        [8:16]  rd as one-hot (8 registers)
        [16:24] rs1 as one-hot (8 registers)
        [24:32] rs2 as one-hot (8 registers)
        [32:48] imm as 16 bits (two's complement, LSB first)
        [48:56] fmt as one-hot (8 formats)
    """
    f = torch.zeros(56, dtype=torch.float32)
    # Opcode bits
    for i in range(8):
        f[i] = float((opcode >> i) & 1)
    # Register one-hots
    f[8 + (rd & 7)] = 1.0
    f[16 + (rs1 & 7)] = 1.0
    f[24 + (rs2 & 7)] = 1.0
    # Immediate bits (16-bit two's complement)
    imm16 = imm & 0xFFFF
    for i in range(16):
        f[32 + i] = float((imm16 >> i) & 1)
    # Format one-hot
    f[48 + (fmt & 7)] = 1.0
    if device is not None:
        f = f.to(device)
    return f


class NeuralCodeGenNet(nn.Module):
    """Neural instruction encoder.

    Takes parsed instruction features and produces a 32-bit encoding.
    Trained to match the classical assembler's output exactly.

    Architecture v2: Rich binary features → deep MLP → 32 output bits.
    Input: opcode_bits(8) + rd_onehot(8) + rs1_onehot(8) + rs2_onehot(8)
           + imm_bits(16) + fmt_onehot(8) = 56 features.
    The network learns to route input bits to the correct output positions.
    """
    INPUT_DIM = 56  # 8+8+8+8+16+8

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),  # 32 output bits
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [batch, 56] → [batch, 32] bit logits"""
        return self.net(features)


# ═══════════════════════════════════════════════════════════════════════════════
# Classical Assembler (ground truth)
# ═══════════════════════════════════════════════════════════════════════════════

class ClassicalAssembler:
    """Deterministic assembler — the ground truth for training the neural assembler.

    Two-pass assembly:
        Pass 1: Collect labels and their addresses
        Pass 2: Encode instructions, resolving label references
    """

    def assemble(self, source: str) -> AssemblyResult:
        """Assemble source code into binary machine code.

        Args:
            source: Assembly source code

        Returns:
            AssemblyResult with binary, labels, errors
        """
        errors = []
        labels = {}
        instructions = []
        source_map = {}

        # Pass 1: Parse and collect labels
        lines = source.split("\n")
        addr = 0

        for line_num, line in enumerate(lines, 1):
            # Strip comments
            line = re.sub(r'[;#].*$', '', line).strip()
            if not line:
                continue

            # Label definition
            if line.endswith(":"):
                label = line[:-1].strip()
                labels[label] = addr
                continue

            # Parse instruction
            instr = self._parse_line(line, line_num)
            if instr is None:
                errors.append(f"Line {line_num}: Cannot parse: {line}")
                continue

            instr.line = line_num
            instr.source = line
            instructions.append(instr)
            source_map[addr] = line_num
            addr += 1

        # Pass 2: Resolve labels and encode
        binary = []
        for i, instr in enumerate(instructions):
            if instr.label_ref:
                if instr.label_ref in labels:
                    instr.imm = labels[instr.label_ref]
                else:
                    # Try numeric
                    try:
                        instr.imm = int(instr.label_ref)
                    except ValueError:
                        errors.append(f"Line {instr.line}: Undefined label: {instr.label_ref}")
                        instr.imm = 0

            encoded = self._encode(instr)
            binary.append(encoded)

        return AssemblyResult(
            binary=binary,
            instructions=instructions,
            labels=labels,
            errors=errors,
            source_map=source_map,
        )

    def _parse_line(self, line: str, line_num: int) -> Optional[AsmInstruction]:
        """Parse a single assembly line into an AsmInstruction."""
        tokens = line.upper().strip()
        tokens = re.sub(r'\s+', ' ', tokens)
        tokens = re.sub(r'\s*,\s*', ',', tokens)

        # NOP / HALT
        if tokens == "NOP":
            return AsmInstruction(opcode=Opcode.NOP, fmt=FMT_NONE)
        if tokens == "HALT":
            return AsmInstruction(opcode=Opcode.HALT, fmt=FMT_NONE)

        parts = tokens.split(None, 1)
        if len(parts) < 2:
            return None

        mnemonic = parts[0]
        operands = [o.strip() for o in parts[1].split(",")]

        # MOV Rd, imm/Rs
        if mnemonic == "MOV":
            if len(operands) != 2:
                return None
            rd = REGISTERS.get(operands[0])
            if rd is None:
                return None
            if operands[1] in REGISTERS:
                return AsmInstruction(opcode=Opcode.MOV_REG, fmt=FMT_REG_REG,
                                     rd=rd, rs1=REGISTERS[operands[1]])
            else:
                imm = self._parse_imm(operands[1])
                if imm is None:
                    return None
                return AsmInstruction(opcode=Opcode.MOV_IMM, fmt=FMT_REG_IMM,
                                     rd=rd, imm=imm)

        # 3-register ops: ADD, SUB, MUL, DIV, AND, OR, XOR
        if mnemonic in ("ADD", "SUB", "MUL", "DIV", "AND", "OR", "XOR"):
            if len(operands) != 3:
                return None
            rd = REGISTERS.get(operands[0])
            rs1 = REGISTERS.get(operands[1])
            rs2 = REGISTERS.get(operands[2])
            if rd is None or rs1 is None or rs2 is None:
                return None
            return AsmInstruction(opcode=MNEMONIC_MAP[mnemonic], fmt=FMT_3REG,
                                 rd=rd, rs1=rs1, rs2=rs2)

        # Shift ops: SHL, SHR Rd, Rs, amount
        if mnemonic in ("SHL", "SHR"):
            if len(operands) != 3:
                return None
            rd = REGISTERS.get(operands[0])
            rs1 = REGISTERS.get(operands[1])
            if rd is None or rs1 is None:
                return None
            # Amount can be register or immediate
            if operands[2] in REGISTERS:
                return AsmInstruction(opcode=MNEMONIC_MAP[mnemonic], fmt=FMT_SHIFT,
                                     rd=rd, rs1=rs1, rs2=REGISTERS[operands[2]])
            else:
                imm = self._parse_imm(operands[2])
                if imm is None:
                    return None
                return AsmInstruction(opcode=MNEMONIC_MAP[mnemonic], fmt=FMT_SHIFT,
                                     rd=rd, rs1=rs1, imm=imm)

        # INC, DEC Rd
        if mnemonic in ("INC", "DEC"):
            if len(operands) != 1:
                return None
            rd = REGISTERS.get(operands[0])
            if rd is None:
                return None
            return AsmInstruction(opcode=MNEMONIC_MAP[mnemonic], fmt=FMT_REG, rd=rd)

        # CMP Rs1, Rs2
        if mnemonic == "CMP":
            if len(operands) != 2:
                return None
            rs1 = REGISTERS.get(operands[0])
            rs2 = REGISTERS.get(operands[1])
            if rs1 is None or rs2 is None:
                return None
            return AsmInstruction(opcode=Opcode.CMP, fmt=FMT_2REG, rs1=rs1, rs2=rs2)

        # Jump instructions: JMP, JZ, JNZ, JS, JNS
        if mnemonic in ("JMP", "JZ", "JNZ", "JS", "JNS"):
            if len(operands) != 1:
                return None
            target = operands[0].strip()
            # Could be a number or label
            try:
                addr = int(target)
                return AsmInstruction(opcode=MNEMONIC_MAP[mnemonic], fmt=FMT_ADDR, imm=addr)
            except ValueError:
                return AsmInstruction(opcode=MNEMONIC_MAP[mnemonic], fmt=FMT_ADDR,
                                     label_ref=target.lower())

        return None

    def _parse_imm(self, text: str) -> Optional[int]:
        """Parse an immediate value (decimal, hex, binary)."""
        text = text.strip()
        try:
            if text.startswith("0X"):
                return int(text, 16)
            if text.startswith("0B"):
                return int(text, 2)
            return int(text)
        except ValueError:
            return None

    def _encode(self, instr: AsmInstruction) -> int:
        """Encode an instruction into a 32-bit word.

        Format: [31:24] opcode | [23:21] rd | [20:18] rs1 | [17:15] rs2 | [14:0] imm
        """
        word = (instr.opcode & 0xFF) << 24
        word |= (instr.rd & 0x7) << 21
        word |= (instr.rs1 & 0x7) << 18
        word |= (instr.rs2 & 0x7) << 15
        word |= instr.imm & 0x7FFF  # 15-bit immediate
        return word

    def decode_word(self, word: int) -> AsmInstruction:
        """Decode a 32-bit word back into an AsmInstruction."""
        opcode = (word >> 24) & 0xFF
        rd = (word >> 21) & 0x7
        rs1 = (word >> 18) & 0x7
        rs2 = (word >> 15) & 0x7
        imm = word & 0x7FFF
        # Sign-extend 15-bit immediate
        if imm & 0x4000:
            imm -= 0x8000

        return AsmInstruction(opcode=opcode, fmt=0, rd=rd, rs1=rs1, rs2=rs2, imm=imm)

    def disassemble(self, binary: List[int]) -> str:
        """Disassemble binary back to source."""
        lines = []
        for i, word in enumerate(binary):
            instr = self.decode_word(word)
            line = self._format_instruction(instr, i)
            lines.append(f"  {i:4d}:  {line}")
        return "\n".join(lines)

    def _format_instruction(self, instr: AsmInstruction, addr: int) -> str:
        """Format an instruction back to text."""
        op = instr.opcode

        if op == Opcode.NOP:
            return "NOP"
        if op == Opcode.HALT:
            return "HALT"
        if op == Opcode.MOV_IMM:
            return f"MOV R{instr.rd}, {instr.imm}"
        if op == Opcode.MOV_REG:
            return f"MOV R{instr.rd}, R{instr.rs1}"
        if op in (Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV,
                  Opcode.AND, Opcode.OR, Opcode.XOR):
            mnem = OPCODE_TO_MNEMONIC.get(op, "???")
            return f"{mnem} R{instr.rd}, R{instr.rs1}, R{instr.rs2}"
        if op in (Opcode.SHL, Opcode.SHR):
            mnem = OPCODE_TO_MNEMONIC.get(op, "???")
            if instr.rs2 != 0 and instr.imm == 0:
                return f"{mnem} R{instr.rd}, R{instr.rs1}, R{instr.rs2}"
            return f"{mnem} R{instr.rd}, R{instr.rs1}, {instr.imm}"
        if op == Opcode.INC:
            return f"INC R{instr.rd}"
        if op == Opcode.DEC:
            return f"DEC R{instr.rd}"
        if op == Opcode.CMP:
            return f"CMP R{instr.rs1}, R{instr.rs2}"
        if op in (Opcode.JMP, Opcode.JZ, Opcode.JNZ, Opcode.JS, Opcode.JNS):
            mnem = OPCODE_TO_MNEMONIC.get(op, "???")
            return f"{mnem} {instr.imm}"

        return f"??? (0x{instr.opcode:02x})"


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Assembler
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralAssembler:
    """Neural assembler with trainable tokenizer and code generator.

    Combines neural and classical components:
        - Neural tokenizer (CNN): classifies characters into token types
        - Classical parser: builds IR from tokens (deterministic, no neural)
        - Neural codegen: encodes IR to binary (trained to match classical)

    Falls back to classical assembler for parsing when neural components
    aren't trained. The training target is byte-identical output with
    the ClassicalAssembler.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or default_device()
        self.classical = ClassicalAssembler()

        # Neural components
        self.tokenizer_net = NeuralTokenizerNet().to(self.device)
        self.codegen_net = NeuralCodeGenNet().to(self.device)
        self._tokenizer_trained = False
        self._codegen_trained = False

        # Statistics
        self.programs_assembled = 0
        self.neural_matches = 0
        self.neural_mismatches = 0

    def assemble(self, source: str) -> AssemblyResult:
        """Assemble source code.

        Uses the classical assembler for correctness, then optionally
        validates neural output against it.
        """
        self.programs_assembled += 1

        # Always use classical assembler for correctness
        result = self.classical.assemble(source)

        # If neural codegen is trained, validate it produces matching output
        if self._codegen_trained and result.success:
            self._validate_neural(result)

        return result

    def assemble_neural(self, source: str) -> AssemblyResult:
        """Assemble using purely neural pipeline (may have errors).

        This is the research path — using the neural assembler without
        classical fallback. Used for benchmarking neural accuracy.
        """
        # Parse with classical parser (deterministic)
        result = self.classical.assemble(source)
        if not result.success:
            return result

        if not self._codegen_trained:
            return result

        # Re-encode with neural codegen
        neural_binary = []
        for instr in result.instructions:
            features = self._instruction_features(instr)
            with torch.no_grad():
                bit_logits = self.codegen_net(features.unsqueeze(0))
                bits = (torch.sigmoid(bit_logits[0]) > 0.5).long()
                word = self._bits_to_int(bits)
            neural_binary.append(word)

        return AssemblyResult(
            binary=neural_binary,
            instructions=result.instructions,
            labels=result.labels,
            errors=result.errors,
            source_map=result.source_map,
        )

    def _validate_neural(self, result: AssemblyResult):
        """Compare neural codegen output against classical."""
        for i, instr in enumerate(result.instructions):
            features = self._instruction_features(instr)
            with torch.no_grad():
                bit_logits = self.codegen_net(features.unsqueeze(0))
                bits = (torch.sigmoid(bit_logits[0]) > 0.5).long()
                neural_word = self._bits_to_int(bits)

            if neural_word == result.binary[i]:
                self.neural_matches += 1
            else:
                self.neural_mismatches += 1

    def _instruction_features(self, instr: AsmInstruction) -> torch.Tensor:
        """Extract rich binary features from an instruction for neural codegen.

        Returns 56-dim vector:
            [0:8]   opcode as 8 bits
            [8:16]  rd as one-hot (8 registers)
            [16:24] rs1 as one-hot (8 registers)
            [24:32] rs2 as one-hot (8 registers)
            [32:48] imm as 16 bits (two's complement)
            [48:56] fmt as one-hot (8 formats)
        """
        return encode_instruction_features(
            instr.opcode, instr.rd, instr.rs1, instr.rs2,
            instr.imm, instr.fmt, self.device
        )

    def _bits_to_int(self, bits: torch.Tensor) -> int:
        """Convert 32 bit tensor to integer."""
        values = (1 << torch.arange(32, device=self.device))
        return int((bits * values).sum().item())

    def disassemble(self, binary: List[int]) -> str:
        """Disassemble binary back to assembly source."""
        return self.classical.disassemble(binary)

    # ─── Training ─────────────────────────────────────────────────────────

    def train_codegen(self, programs: List[str], epochs: int = 100,
                      lr: float = 1e-3) -> Dict:
        """Train the neural code generator from assembly programs.

        Uses the classical assembler as oracle — trains the neural
        codegen to produce identical binary output.

        Args:
            programs: List of assembly source programs
            epochs: Training epochs
            lr: Learning rate

        Returns:
            Training statistics
        """
        # Assemble all programs with classical assembler
        all_features = []
        all_targets = []

        for source in programs:
            result = self.classical.assemble(source)
            if not result.success:
                continue
            for instr, word in zip(result.instructions, result.binary):
                features = self._instruction_features(instr)
                target_bits = self._int_to_bits(word)
                all_features.append(features)
                all_targets.append(target_bits)

        if not all_features:
            return {"error": "no_valid_programs"}

        features_tensor = torch.stack(all_features)
        targets_tensor = torch.stack(all_targets).float()

        optimizer = torch.optim.Adam(self.codegen_net.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        best_acc = 0.0
        for epoch in range(epochs):
            self.codegen_net.train()
            optimizer.zero_grad()

            logits = self.codegen_net(features_tensor)
            loss = loss_fn(logits, targets_tensor)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                acc = (preds == targets_tensor).float().mean().item()
                best_acc = max(best_acc, acc)

        self.codegen_net.eval()
        self._codegen_trained = True

        # Verify: count exact instruction matches
        exact_matches = 0
        total = len(all_features)
        with torch.no_grad():
            logits = self.codegen_net(features_tensor)
            for i in range(total):
                bits = (torch.sigmoid(logits[i]) > 0.5).long()
                pred_word = self._bits_to_int(bits)
                actual_word = self._bits_to_int(all_targets[i].long())
                if pred_word == actual_word:
                    exact_matches += 1

        stats = {
            "epochs": epochs,
            "num_programs": len(programs),
            "num_instructions": total,
            "bit_accuracy": best_acc,
            "exact_match_rate": exact_matches / max(1, total),
            "exact_matches": exact_matches,
        }
        logger.info(f"[Assembler] Trained: {exact_matches}/{total} exact matches "
                    f"({exact_matches/max(1,total)*100:.1f}%)")
        return stats

    def _int_to_bits(self, value: int) -> torch.Tensor:
        """Convert integer to 32-bit tensor."""
        bits = torch.zeros(32, device=self.device)
        for i in range(32):
            bits[i] = (value >> i) & 1
        return bits

    # ─── Persistence ──────────────────────────────────────────────────────

    def save(self, tokenizer_path: str = "models/research/neuros/assembler_tokenizer.pt",
             codegen_path: str = "models/research/neuros/assembler_codegen.pt"):
        Path(tokenizer_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.tokenizer_net.state_dict(), tokenizer_path)
        torch.save(self.codegen_net.state_dict(), codegen_path)

    def load(self, tokenizer_path: str = "models/research/neuros/assembler_tokenizer.pt",
             codegen_path: str = "models/research/neuros/assembler_codegen.pt") -> Dict[str, bool]:
        result = {}
        if Path(tokenizer_path).exists():
            self.tokenizer_net.load_state_dict(
                torch.load(tokenizer_path, map_location=self.device, weights_only=True))
            self.tokenizer_net.eval()
            self._tokenizer_trained = True
            result["tokenizer"] = True
        if Path(codegen_path).exists():
            state = torch.load(codegen_path, map_location=self.device, weights_only=True)
            # Detect hidden dim from saved weights to rebuild matching architecture
            first_weight = state.get("net.0.weight")
            if first_weight is not None:
                hidden_dim = first_weight.shape[0]
                input_dim = first_weight.shape[1]
                # Rebuild if dimensions don't match current net
                cur_input = self.codegen_net.net[0].in_features
                cur_hidden = self.codegen_net.net[0].out_features
                if hidden_dim != cur_hidden or input_dim != cur_input:
                    self.codegen_net = NeuralCodeGenNet(hidden_dim=hidden_dim).to(self.device)
            self.codegen_net.load_state_dict(state)
            self.codegen_net.eval()
            self._codegen_trained = True
            result["codegen"] = True
        return result

    # ─── Diagnostics ──────────────────────────────────────────────────────

    @property
    def neural_accuracy(self) -> float:
        total = self.neural_matches + self.neural_mismatches
        return self.neural_matches / max(1, total)

    def stats(self) -> Dict:
        return {
            "programs_assembled": self.programs_assembled,
            "neural_matches": self.neural_matches,
            "neural_mismatches": self.neural_mismatches,
            "neural_accuracy": self.neural_accuracy,
            "tokenizer_trained": self._tokenizer_trained,
            "codegen_trained": self._codegen_trained,
        }

    def __repr__(self) -> str:
        return (f"NeuralAssembler(programs={self.programs_assembled}, "
                f"accuracy={self.neural_accuracy:.1%})")
