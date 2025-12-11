"""DecodeLLM: Semantic instruction decoder for KVRM-CPU.

This module implements the core KVRM innovation for CPU emulation:
replacing hardcoded silicon decode logic with semantic LLM-based
instruction understanding that emits verified registry keys.

Architecture:
    Raw instruction → DecodeLLM → (operation_key, params) → Registry → Execute

Modes:
    - Mock Mode: Rule-based regex parsing (for development/testing)
    - Real Mode: Fine-tuned micro-LLM for semantic understanding

The decoder takes natural assembly language and produces structured
operation keys that the verified registry can execute safely.
"""

import re
import json
from typing import Dict, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class DecodeResult:
    """Result of instruction decode operation.

    Attributes:
        key: Operation key (e.g., "OP_ADD")
        params: Operation parameters dictionary
        valid: Whether decode succeeded
        error: Error message if decode failed
        raw_instruction: Original instruction string
    """
    key: str
    params: Dict
    valid: bool
    error: Optional[str] = None
    raw_instruction: str = ""


class DecodeLLM:
    """Semantic instruction decoder using LLM or mock rules.

    This is the heart of the KVRM-CPU paradigm: semantic understanding
    of instructions rather than hardcoded bit-pattern matching.

    Attributes:
        mock_mode: Whether to use rule-based mock decoder
        model_path: Path to trained micro-LLM (if not mock mode)
        valid_keys: Set of valid operation keys
        labels: Dictionary mapping label names to addresses
    """

    # Valid operation keys that can be emitted
    VALID_KEYS: Set[str] = {
        "OP_MOV_REG_IMM",
        "OP_MOV_REG_REG",
        "OP_ADD",
        "OP_SUB",
        "OP_MUL",
        "OP_INC",
        "OP_DEC",
        "OP_CMP",
        "OP_JMP",
        "OP_JZ",
        "OP_JNZ",
        "OP_JS",
        "OP_JNS",
        "OP_HALT",
        "OP_NOP",
        "OP_INVALID",
    }

    # Valid register names
    REGISTERS: Set[str] = {"R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"}

    def __init__(self, mock_mode: bool = True, model_path: Optional[str] = None):
        """Initialize the decoder.

        Args:
            mock_mode: Use rule-based decoder (True) or LLM (False)
            model_path: Path to trained model (required if mock_mode=False)
        """
        self.mock_mode = mock_mode
        self.model_path = model_path
        self.labels: Dict[str, int] = {}
        self._model = None
        self._tokenizer = None

        if not mock_mode and model_path is None:
            raise ValueError("model_path required when mock_mode=False")

    def set_labels(self, labels: Dict[str, int]) -> None:
        """Set label-to-address mapping for jump resolution.

        Args:
            labels: Dictionary mapping label names to instruction addresses
        """
        self.labels = labels

    def load(self) -> None:
        """Load the model (only needed for real mode)."""
        if self.mock_mode:
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            # Try loading as PEFT model first
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    torch_dtype=torch.float32,
                    device_map=None
                )
                self._model = PeftModel.from_pretrained(base_model, self.model_path)
            except Exception:
                # Fall back to direct load
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    device_map=None
                )

            # Move to appropriate device
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._model = self._model.to(device)
            self._model.eval()

        except ImportError:
            raise ImportError("Real mode requires torch and transformers")

    def unload(self) -> None:
        """Unload the model to free memory."""
        self._model = None
        self._tokenizer = None

    def decode(self, instruction: str) -> DecodeResult:
        """Decode an instruction to operation key and parameters.

        Args:
            instruction: Assembly instruction string (e.g., "ADD R3, R1, R2")

        Returns:
            DecodeResult with operation key and parameters
        """
        instruction = instruction.strip()

        if not instruction:
            return DecodeResult(
                key="OP_INVALID",
                params={"raw": ""},
                valid=False,
                error="Empty instruction",
                raw_instruction=""
            )

        if self.mock_mode:
            return self._mock_decode(instruction)
        else:
            return self._llm_decode(instruction)

    # =========================================================================
    # Mock Mode (Rule-Based Decoder)
    # =========================================================================

    def _mock_decode(self, instruction: str) -> DecodeResult:
        """Rule-based instruction decode using regex patterns.

        This simulates what the trained LLM will do, but with
        deterministic pattern matching for development/testing.
        """
        # Normalize: uppercase, remove extra whitespace
        instr = instruction.upper().strip()
        instr = re.sub(r'\s+', ' ', instr)
        instr = re.sub(r'\s*,\s*', ',', instr)  # Normalize comma spacing

        # Try each instruction pattern
        try:
            # HALT
            if instr == "HALT":
                return DecodeResult("OP_HALT", {}, True, raw_instruction=instruction)

            # NOP
            if instr == "NOP":
                return DecodeResult("OP_NOP", {}, True, raw_instruction=instruction)

            # MOV Rd, imm or MOV Rd, Rs
            mov_match = re.match(r'^MOV\s+(R[0-7])[,\s]+(.+)$', instr)
            if mov_match:
                dest = mov_match.group(1)
                src = mov_match.group(2).strip()

                # Check if source is register or immediate
                if src in self.REGISTERS:
                    return DecodeResult(
                        "OP_MOV_REG_REG",
                        {"dest": dest, "src": src},
                        True,
                        raw_instruction=instruction
                    )
                else:
                    # Try parsing as integer
                    try:
                        value = self._parse_immediate(src)
                        return DecodeResult(
                            "OP_MOV_REG_IMM",
                            {"dest": dest, "value": value},
                            True,
                            raw_instruction=instruction
                        )
                    except ValueError:
                        return DecodeResult(
                            "OP_INVALID",
                            {"raw": instruction},
                            False,
                            error=f"Invalid MOV source: {src}",
                            raw_instruction=instruction
                        )

            # ADD Rd, Rs1, Rs2
            add_match = re.match(r'^ADD\s+(R[0-7])[,\s]+(R[0-7])[,\s]+(R[0-7])$', instr)
            if add_match:
                return DecodeResult(
                    "OP_ADD",
                    {
                        "dest": add_match.group(1),
                        "src1": add_match.group(2),
                        "src2": add_match.group(3)
                    },
                    True,
                    raw_instruction=instruction
                )

            # SUB Rd, Rs1, Rs2
            sub_match = re.match(r'^SUB\s+(R[0-7])[,\s]+(R[0-7])[,\s]+(R[0-7])$', instr)
            if sub_match:
                return DecodeResult(
                    "OP_SUB",
                    {
                        "dest": sub_match.group(1),
                        "src1": sub_match.group(2),
                        "src2": sub_match.group(3)
                    },
                    True,
                    raw_instruction=instruction
                )

            # MUL Rd, Rs1, Rs2
            mul_match = re.match(r'^MUL\s+(R[0-7])[,\s]+(R[0-7])[,\s]+(R[0-7])$', instr)
            if mul_match:
                return DecodeResult(
                    "OP_MUL",
                    {
                        "dest": mul_match.group(1),
                        "src1": mul_match.group(2),
                        "src2": mul_match.group(3)
                    },
                    True,
                    raw_instruction=instruction
                )

            # INC Rd
            inc_match = re.match(r'^INC\s+(R[0-7])$', instr)
            if inc_match:
                return DecodeResult(
                    "OP_INC",
                    {"dest": inc_match.group(1)},
                    True,
                    raw_instruction=instruction
                )

            # DEC Rd
            dec_match = re.match(r'^DEC\s+(R[0-7])$', instr)
            if dec_match:
                return DecodeResult(
                    "OP_DEC",
                    {"dest": dec_match.group(1)},
                    True,
                    raw_instruction=instruction
                )

            # CMP Rs1, Rs2
            cmp_match = re.match(r'^CMP\s+(R[0-7])[,\s]+(R[0-7])$', instr)
            if cmp_match:
                return DecodeResult(
                    "OP_CMP",
                    {
                        "src1": cmp_match.group(1),
                        "src2": cmp_match.group(2)
                    },
                    True,
                    raw_instruction=instruction
                )

            # JMP addr/label
            jmp_match = re.match(r'^JMP\s+(.+)$', instr)
            if jmp_match:
                target = jmp_match.group(1).strip()
                addr = self._resolve_address(target)
                if addr is not None:
                    return DecodeResult(
                        "OP_JMP",
                        {"addr": addr},
                        True,
                        raw_instruction=instruction
                    )
                else:
                    return DecodeResult(
                        "OP_INVALID",
                        {"raw": instruction},
                        False,
                        error=f"Unknown label: {target}",
                        raw_instruction=instruction
                    )

            # JZ addr/label
            jz_match = re.match(r'^JZ\s+(.+)$', instr)
            if jz_match:
                target = jz_match.group(1).strip()
                addr = self._resolve_address(target)
                if addr is not None:
                    return DecodeResult(
                        "OP_JZ",
                        {"addr": addr},
                        True,
                        raw_instruction=instruction
                    )
                else:
                    return DecodeResult(
                        "OP_INVALID",
                        {"raw": instruction},
                        False,
                        error=f"Unknown label: {target}",
                        raw_instruction=instruction
                    )

            # JNZ addr/label
            jnz_match = re.match(r'^JNZ\s+(.+)$', instr)
            if jnz_match:
                target = jnz_match.group(1).strip()
                addr = self._resolve_address(target)
                if addr is not None:
                    return DecodeResult(
                        "OP_JNZ",
                        {"addr": addr},
                        True,
                        raw_instruction=instruction
                    )
                else:
                    return DecodeResult(
                        "OP_INVALID",
                        {"raw": instruction},
                        False,
                        error=f"Unknown label: {target}",
                        raw_instruction=instruction
                    )

            # JS addr/label (jump if sign flag set / negative)
            js_match = re.match(r'^JS\s+(.+)$', instr)
            if js_match:
                target = js_match.group(1).strip()
                addr = self._resolve_address(target)
                if addr is not None:
                    return DecodeResult(
                        "OP_JS",
                        {"addr": addr},
                        True,
                        raw_instruction=instruction
                    )
                else:
                    return DecodeResult(
                        "OP_INVALID",
                        {"raw": instruction},
                        False,
                        error=f"Unknown label: {target}",
                        raw_instruction=instruction
                    )

            # JNS addr/label (jump if sign flag not set / non-negative)
            jns_match = re.match(r'^JNS\s+(.+)$', instr)
            if jns_match:
                target = jns_match.group(1).strip()
                addr = self._resolve_address(target)
                if addr is not None:
                    return DecodeResult(
                        "OP_JNS",
                        {"addr": addr},
                        True,
                        raw_instruction=instruction
                    )
                else:
                    return DecodeResult(
                        "OP_INVALID",
                        {"raw": instruction},
                        False,
                        error=f"Unknown label: {target}",
                        raw_instruction=instruction
                    )

            # Unknown instruction
            return DecodeResult(
                "OP_INVALID",
                {"raw": instruction},
                False,
                error=f"Unknown instruction format: {instruction}",
                raw_instruction=instruction
            )

        except Exception as e:
            return DecodeResult(
                "OP_INVALID",
                {"raw": instruction},
                False,
                error=str(e),
                raw_instruction=instruction
            )

    def _parse_immediate(self, value: str) -> int:
        """Parse an immediate value (decimal, hex, or binary).

        Args:
            value: String representation of number

        Returns:
            Integer value

        Raises:
            ValueError: If value cannot be parsed
        """
        value = value.strip().upper()

        # Hex: 0x prefix
        if value.startswith("0X"):
            return int(value, 16)

        # Binary: 0b prefix
        if value.startswith("0B"):
            return int(value, 2)

        # Decimal (may be negative)
        return int(value)

    def _resolve_address(self, target: str) -> Optional[int]:
        """Resolve a jump target to an address.

        Args:
            target: Label name or numeric address

        Returns:
            Address as integer, or None if unresolvable
        """
        # First try as numeric address
        try:
            return int(target)
        except ValueError:
            pass

        # Try as label (case-insensitive lookup)
        target_upper = target.upper()
        for label, addr in self.labels.items():
            if label.upper() == target_upper:
                return addr

        return None

    # =========================================================================
    # Real Mode (LLM-Based Decoder)
    # =========================================================================

    def _llm_decode(self, instruction: str) -> DecodeResult:
        """LLM-based semantic instruction decode.

        Uses the fine-tuned micro-LLM to understand the instruction
        and emit a structured operation key with parameters.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        # Format prompt
        prompt = f"### Context:\n{instruction}\n\n### Key:\n"

        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        # Move to device
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # Decode output
        generated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the key portion after "### Key:\n"
        if "### Key:" in generated:
            key_text = generated.split("### Key:")[-1].strip()
        else:
            key_text = generated.strip()

        # Parse JSON
        try:
            # Find JSON object
            json_match = re.search(r'\{[^}]+\}', key_text)
            if json_match:
                result = json.loads(json_match.group())
                key = result.get("key", "OP_INVALID")
                params = {k: v for k, v in result.items() if k != "key"}

                # Validate key
                if key not in self.VALID_KEYS:
                    return DecodeResult(
                        "OP_INVALID",
                        {"raw": instruction},
                        False,
                        error=f"Invalid key from LLM: {key}",
                        raw_instruction=instruction
                    )

                # Resolve label addresses in jump params
                if key in ("OP_JMP", "OP_JZ", "OP_JNZ", "OP_JS", "OP_JNS") and "addr" in params:
                    addr = params["addr"]
                    if isinstance(addr, str):
                        resolved = self._resolve_address(addr)
                        if resolved is not None:
                            params["addr"] = resolved
                        else:
                            return DecodeResult(
                                "OP_INVALID",
                                {"raw": instruction},
                                False,
                                error=f"Unknown label: {addr}",
                                raw_instruction=instruction
                            )

                return DecodeResult(key, params, True, raw_instruction=instruction)
            else:
                return DecodeResult(
                    "OP_INVALID",
                    {"raw": instruction},
                    False,
                    error=f"No JSON found in LLM output: {key_text}",
                    raw_instruction=instruction
                )

        except json.JSONDecodeError as e:
            return DecodeResult(
                "OP_INVALID",
                {"raw": instruction},
                False,
                error=f"JSON parse error: {e}",
                raw_instruction=instruction
            )


def parse_program(source: str) -> Tuple[list, Dict[str, int]]:
    """Parse assembly source code into instructions and labels.

    Handles:
        - Labels (lines ending with :)
        - Comments (starting with ; or #)
        - Blank lines

    Args:
        source: Assembly source code

    Returns:
        Tuple of (list of instruction strings, label-to-address dict)
    """
    instructions = []
    labels = {}

    for line in source.split("\n"):
        # Remove comments
        line = re.sub(r'[;#].*$', '', line).strip()

        if not line:
            continue

        # Check for label
        if line.endswith(":"):
            label = line[:-1].strip()
            labels[label] = len(instructions)
        else:
            instructions.append(line)

    return instructions, labels
