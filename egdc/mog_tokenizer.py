"""Byte-level tokenizer for Mog code generation.

Same approach as python_tokenizer.py: each byte (0-255) is a token, plus 4 special tokens.
Mog is ASCII so byte-level tokenization is clean and simple.

Vocabulary layout (260 tokens):
    0-255:  raw byte values
    256:    MASK  (diffusion masking token)
    257:    PAD   (sequence padding)
    258:    BOS   (beginning of sequence)
    259:    EOS   (end of sequence)
"""

from __future__ import annotations
from typing import List


# --- Vocabulary constants ---------------------------------------------------

NUM_BYTES = 256

MASK_TOKEN = 256
PAD_TOKEN = 257
BOS_TOKEN = 258
EOS_TOKEN = 259
VOCAB_SIZE = 260

SPECIAL_TOKENS = {MASK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN}


class MogCodeTokenizer:
    """Byte-level tokenizer for Mog source code.

    Each character is encoded as its UTF-8 byte(s). Mog is ASCII so
    each character maps to exactly one token. Special tokens are appended
    for diffusion control (MASK, PAD, BOS, EOS).
    """

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

    def encode(self, code: str, add_bos_eos: bool = True) -> List[int]:
        """Encode a Mog code string into byte-level token IDs.

        Args:
            code: Mog source code string.
            add_bos_eos: Whether to prepend BOS and append EOS.

        Returns:
            List of integer token IDs.
        """
        raw_bytes = code.encode("utf-8")
        tokens: List[int] = []
        if add_bos_eos:
            tokens.append(BOS_TOKEN)
        tokens.extend(raw_bytes)
        if add_bos_eos:
            tokens.append(EOS_TOKEN)
        return tokens

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs back into a Mog code string.

        Args:
            token_ids: List of integer token IDs.
            skip_special: If True, BOS/EOS/PAD/MASK tokens are skipped.

        Returns:
            Decoded Mog source code string.
        """
        byte_values: List[int] = []
        for tid in token_ids:
            if skip_special and tid in SPECIAL_TOKENS:
                continue
            if 0 <= tid < NUM_BYTES:
                byte_values.append(tid)
            # Out-of-range non-special tokens are silently dropped
        return bytes(byte_values).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def pad(self, token_ids: List[int], length: int) -> List[int]:
        """Pad or truncate token_ids to exactly `length`."""
        if len(token_ids) >= length:
            return token_ids[:length]
        return token_ids + [PAD_TOKEN] * (length - len(token_ids))

    def token_count(self, token_ids: List[int]) -> int:
        """Count non-special tokens."""
        return sum(1 for t in token_ids if t not in SPECIAL_TOKENS)
