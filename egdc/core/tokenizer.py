"""nCPU ISA Tokenizer.

Each instruction is encoded as exactly 4 tokens:
    [opcode] [dst_reg] [src_reg] [imm_or_branch_target]

For instructions that don't use all fields, unused slots are filled with R0/IMM_0.

Vocabulary layout (346 tokens total):
    0-13:    14 opcodes (NOP, MOV_IMM, MOV_REG, ADD, SUB, MUL, AND, OR, XOR,
                         CMP, BEQ, BNE, BGT, HALT)
    14-21:   8 registers (R0..R7)
    22-277:  256 immediate values (IMM_0..IMM_255)
    278-341: 64 branch targets (BR_0..BR_63)
    342:     MASK  (diffusion masking token)
    343:     PAD   (sequence padding)
    344:     BOS   (beginning of sequence)
    345:     EOS   (end of sequence)
"""

from __future__ import annotations
from typing import List, Optional


# --- Vocabulary constants ---------------------------------------------------

OPCODES = [
    "NOP", "MOV_IMM", "MOV_REG", "ADD", "SUB", "MUL",
    "AND", "OR", "XOR", "CMP", "BEQ", "BNE", "BGT", "HALT",
]
NUM_OPCODES = 14

REGISTERS = [f"R{i}" for i in range(8)]
NUM_REGISTERS = 8

NUM_IMMEDIATES = 256
NUM_BRANCH_TARGETS = 64

# Token-id offsets
OPCODE_OFFSET = 0          # 0..13
REG_OFFSET = NUM_OPCODES   # 14..21
IMM_OFFSET = REG_OFFSET + NUM_REGISTERS          # 22..277
BR_OFFSET = IMM_OFFSET + NUM_IMMEDIATES          # 278..341
MASK_TOKEN = BR_OFFSET + NUM_BRANCH_TARGETS      # 342
PAD_TOKEN = MASK_TOKEN + 1                        # 343
BOS_TOKEN = PAD_TOKEN + 1                         # 344
EOS_TOKEN = BOS_TOKEN + 1                         # 345
VOCAB_SIZE = EOS_TOKEN + 1                        # 346

# Which opcodes use which operand slots
# Format: (uses_dst_reg, uses_src_reg, uses_imm, uses_branch)
_OPCODE_SLOTS = {
    "NOP":     (False, False, False, False),
    "MOV_IMM": (True,  False, True,  False),
    "MOV_REG": (True,  True,  False, False),
    "ADD":     (True,  True,  False, False),  # dst = dst + src
    "SUB":     (True,  True,  False, False),
    "MUL":     (True,  True,  False, False),
    "AND":     (True,  True,  False, False),
    "OR":      (True,  True,  False, False),
    "XOR":     (True,  True,  False, False),
    "CMP":     (True,  True,  False, False),  # dst compared with src
    "BEQ":     (False, False, False, True),
    "BNE":     (False, False, False, True),
    "BGT":     (False, False, False, True),
    "HALT":    (False, False, False, False),
}

# Reverse maps
_OPCODE_TO_ID = {name: i for i, name in enumerate(OPCODES)}
_REG_TO_ID = {name: REG_OFFSET + i for i, name in enumerate(REGISTERS)}
_ID_TO_OPCODE = {i: name for name, i in _OPCODE_TO_ID.items()}
_ID_TO_REG = {v: k for k, v in _REG_TO_ID.items()}


class NCPUTokenizer:
    """Tokenizer for the nCPU 14-opcode ISA.

    Encodes assembly text to token ids and decodes back.
    """

    # Expose constants as class attributes for convenience
    MASK = MASK_TOKEN
    PAD = PAD_TOKEN
    BOS = BOS_TOKEN
    EOS = EOS_TOKEN

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, program_text: str, add_bos_eos: bool = True) -> List[int]:
        """Encode assembly text into a list of token ids.

        Each instruction becomes exactly 4 tokens. Unused operand slots are
        filled with default values (R0 for registers, IMM_0 for immediates).

        Args:
            program_text: Multi-line assembly, one instruction per line.
                          Blank lines and '#' comments are ignored.
            add_bos_eos: Whether to prepend BOS and append EOS.

        Returns:
            List of integer token ids.
        """
        tokens: List[int] = []
        if add_bos_eos:
            tokens.append(BOS_TOKEN)

        for line in program_text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Strip inline comments
            if "#" in line:
                line = line[: line.index("#")].strip()
            parts = line.split()
            opcode_str = parts[0].upper()
            if opcode_str not in _OPCODE_TO_ID:
                raise ValueError(f"Unknown opcode: {opcode_str}")

            opcode_id = _OPCODE_TO_ID[opcode_str]
            slots = _OPCODE_SLOTS[opcode_str]
            uses_dst, uses_src, uses_imm, uses_br = slots

            # Defaults
            dst_id = REG_OFFSET  # R0
            src_id = REG_OFFSET  # R0
            imm_id = IMM_OFFSET  # IMM_0

            operands = parts[1:]

            if opcode_str == "HALT" or opcode_str == "NOP":
                # No operands
                pass
            elif opcode_str == "MOV_IMM":
                # MOV_IMM Rd <imm>
                dst_id = self._parse_reg(operands[0])
                imm_id = self._parse_imm(operands[1])
            elif opcode_str == "MOV_REG":
                # MOV_REG Rd Rs
                dst_id = self._parse_reg(operands[0])
                src_id = self._parse_reg(operands[1])
            elif opcode_str in ("ADD", "SUB", "MUL", "AND", "OR", "XOR"):
                # OP Rd Rs  (Rd = Rd op Rs)  — 2-operand form
                # OP Rd Rs1 Rs2 — not in this ISA, but handle 2-op
                dst_id = self._parse_reg(operands[0])
                src_id = self._parse_reg(operands[1])
            elif opcode_str == "CMP":
                # CMP Ra Rb
                dst_id = self._parse_reg(operands[0])
                src_id = self._parse_reg(operands[1])
            elif opcode_str in ("BEQ", "BNE", "BGT"):
                # Branch <target>
                imm_id = self._parse_branch(operands[0])
            else:
                raise ValueError(f"Unhandled opcode: {opcode_str}")

            tokens.extend([opcode_id, dst_id, src_id, imm_id])

        if add_bos_eos:
            tokens.append(EOS_TOKEN)

        return tokens

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Decode token ids back into assembly text.

        Args:
            token_ids: List of integer token ids.
            skip_special: If True, BOS/EOS/PAD/MASK tokens are skipped.

        Returns:
            Multi-line assembly string.
        """
        lines: List[str] = []
        ids = list(token_ids)

        # Strip special tokens at boundaries
        if skip_special:
            ids = [t for t in ids if t not in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, MASK_TOKEN)]

        if len(ids) % 4 != 0:
            # Best-effort: truncate to multiple of 4
            ids = ids[: (len(ids) // 4) * 4]

        for i in range(0, len(ids), 4):
            opcode_id, dst_id, src_id, imm_id = ids[i : i + 4]
            opcode_str = _ID_TO_OPCODE.get(opcode_id, f"UNK_{opcode_id}")
            slots = _OPCODE_SLOTS.get(opcode_str, (False, False, False, False))
            uses_dst, uses_src, uses_imm, uses_br = slots

            if opcode_str in ("NOP", "HALT"):
                lines.append(opcode_str)
            elif opcode_str == "MOV_IMM":
                reg = self._id_to_reg(dst_id)
                imm = self._id_to_imm(imm_id)
                lines.append(f"MOV_IMM {reg} {imm}")
            elif opcode_str == "MOV_REG":
                dst = self._id_to_reg(dst_id)
                src = self._id_to_reg(src_id)
                lines.append(f"MOV_REG {dst} {src}")
            elif opcode_str in ("ADD", "SUB", "MUL", "AND", "OR", "XOR"):
                dst = self._id_to_reg(dst_id)
                src = self._id_to_reg(src_id)
                lines.append(f"{opcode_str} {dst} {src}")
            elif opcode_str == "CMP":
                dst = self._id_to_reg(dst_id)
                src = self._id_to_reg(src_id)
                lines.append(f"CMP {dst} {src}")
            elif opcode_str in ("BEQ", "BNE", "BGT"):
                br = self._id_to_branch(imm_id)
                lines.append(f"{opcode_str} {br}")
            else:
                lines.append(f"# unknown opcode_id={opcode_id}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_reg(s: str) -> int:
        s = s.upper().rstrip(",")
        if s not in _REG_TO_ID:
            raise ValueError(f"Unknown register: {s}")
        return _REG_TO_ID[s]

    @staticmethod
    def _parse_imm(s: str) -> int:
        val = int(s)
        if not (0 <= val <= 255):
            raise ValueError(f"Immediate out of range: {val}")
        return IMM_OFFSET + val

    @staticmethod
    def _parse_branch(s: str) -> int:
        val = int(s)
        if not (0 <= val <= 63):
            raise ValueError(f"Branch target out of range: {val}")
        return BR_OFFSET + val

    @staticmethod
    def _id_to_reg(tid: int) -> str:
        idx = tid - REG_OFFSET
        if 0 <= idx < NUM_REGISTERS:
            return f"R{idx}"
        return f"R?({tid})"

    @staticmethod
    def _id_to_imm(tid: int) -> str:
        idx = tid - IMM_OFFSET
        if 0 <= idx < NUM_IMMEDIATES:
            return str(idx)
        # Maybe it's a branch target encoded in imm slot
        bidx = tid - BR_OFFSET
        if 0 <= bidx < NUM_BRANCH_TARGETS:
            return str(bidx)
        return f"?({tid})"

    @staticmethod
    def _id_to_branch(tid: int) -> str:
        idx = tid - BR_OFFSET
        if 0 <= idx < NUM_BRANCH_TARGETS:
            return str(idx)
        # Fallback: might be an immediate in the branch slot
        iidx = tid - IMM_OFFSET
        if 0 <= iidx < NUM_IMMEDIATES:
            return str(iidx)
        return f"?({tid})"

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def pad(self, token_ids: List[int], length: int) -> List[int]:
        """Pad or truncate token_ids to exactly `length`."""
        if len(token_ids) >= length:
            return token_ids[:length]
        return token_ids + [PAD_TOKEN] * (length - len(token_ids))

    def instruction_count(self, token_ids: List[int]) -> int:
        """Count instructions (excluding BOS/EOS/PAD)."""
        clean = [t for t in token_ids if t not in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, MASK_TOKEN)]
        return len(clean) // 4
