"""Constant-time primitive operations.

Every function in this module executes in time independent of its inputs.
No data-dependent branches, no variable-time lookups, no cache-timing leaks.
Implemented as pure tensor operations for GPU execution.

Design Principles
-----------------
1. **No branches**: All control flow is replaced with arithmetic masking.
   ``ct_select(cond, a, b)`` replaces ``a if cond else b``.

2. **No early termination**: ``ct_memcmp`` compares ALL bytes even if the
   first byte differs. ``ct_byte_lookup`` scans the ENTIRE table even if
   the match is at index 0.

3. **No variable-time indexing**: Table lookups use full linear scans
   (O(n) regardless of index) rather than direct indexing which would
   leak the index through cache access patterns on conventional CPUs.

4. **Bitwise operations via bit decomposition**: Since PyTorch float tensors
   do not support bitwise XOR/AND/OR directly, we decompose byte values into
   8-bit vectors, perform operations on individual bits, and recombine. This
   is naturally constant-time: always exactly 8 bits processed regardless of
   input values.

On nCPU's GPU, these properties are verified by measuring cycle counts across
different inputs: sigma=0.0 confirms constant-time execution.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Bit decomposition for float-tensor bitwise operations
# ---------------------------------------------------------------------------

def _to_bits(x: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """Decompose byte values (0-255 as floats) into individual bits.

    Args:
        x: tensor of float values representing bytes (each in [0, 255])
        n_bits: number of bits to extract (default 8 for bytes)

    Returns:
        Tensor with an extra trailing dimension of size n_bits.
        Bit 0 is the least significant bit.
    """
    # Create power-of-2 divisors: [1, 2, 4, 8, 16, 32, 64, 128]
    powers = (2.0 ** torch.arange(n_bits, dtype=x.dtype, device=x.device))
    # Integer division then mod 2 extracts each bit
    # floor(x / 2^i) mod 2
    x_expanded = x.unsqueeze(-1)
    bits = torch.floor(x_expanded / powers) % 2.0
    return bits


def _from_bits(bits: torch.Tensor) -> torch.Tensor:
    """Recombine individual bits back into byte values.

    Args:
        bits: tensor with trailing dimension of size n_bits (bit values 0.0 or 1.0)

    Returns:
        Tensor with the trailing dimension collapsed, containing byte values.
    """
    n_bits = bits.shape[-1]
    powers = (2.0 ** torch.arange(n_bits, dtype=bits.dtype, device=bits.device))
    return (bits * powers).sum(dim=-1)


# ---------------------------------------------------------------------------
# Bitwise operations on float byte tensors
# ---------------------------------------------------------------------------

def ct_xor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Constant-time bitwise XOR on float tensors representing bytes.

    Decomposes into bits, XORs each bit position via arithmetic
    (XOR = a + b - 2*a*b for bits in {0,1}), then recombines.
    Always processes all 8 bits regardless of input values.

    Args:
        a: float tensor with values in [0, 255]
        b: float tensor with values in [0, 255]

    Returns:
        Float tensor with XOR result, values in [0, 255]
    """
    bits_a = _to_bits(a)
    bits_b = _to_bits(b)
    # XOR for binary: a ^ b = a + b - 2*a*b
    xor_bits = bits_a + bits_b - 2.0 * bits_a * bits_b
    return _from_bits(xor_bits)


def ct_and(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Constant-time bitwise AND on float tensors representing bytes.

    Args:
        a: float tensor with values in [0, 255]
        b: float tensor with values in [0, 255]

    Returns:
        Float tensor with AND result, values in [0, 255]
    """
    bits_a = _to_bits(a)
    bits_b = _to_bits(b)
    # AND for binary: a & b = a * b
    and_bits = bits_a * bits_b
    return _from_bits(and_bits)


def ct_or(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Constant-time bitwise OR on float tensors representing bytes.

    Args:
        a: float tensor with values in [0, 255]
        b: float tensor with values in [0, 255]

    Returns:
        Float tensor with OR result, values in [0, 255]
    """
    bits_a = _to_bits(a)
    bits_b = _to_bits(b)
    # OR for binary: a | b = a + b - a*b
    or_bits = bits_a + bits_b - bits_a * bits_b
    return _from_bits(or_bits)


def ct_not(a: torch.Tensor) -> torch.Tensor:
    """Constant-time bitwise NOT (8-bit) on float tensors representing bytes.

    Computes ~a as 255 - a (one's complement for 8-bit values).

    Args:
        a: float tensor with values in [0, 255]

    Returns:
        Float tensor with NOT result, values in [0, 255]
    """
    return 255.0 - a


# ---------------------------------------------------------------------------
# Core constant-time primitives
# ---------------------------------------------------------------------------

def ct_select(condition: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Constant-time conditional select: returns a if condition else b.

    No branching -- uses arithmetic masking. Both a and b are always read
    and both multiplications always execute, regardless of condition.

    This is the fundamental building block for branchless programming.
    Every ``if/else`` in a cryptographic algorithm should be replaced
    with ct_select to eliminate timing leaks.

    Args:
        condition: tensor of 0.0 or 1.0 values (the selector)
        a: values to return where condition is 1.0
        b: values to return where condition is 0.0

    Returns:
        Element-wise: a[i] if condition[i] == 1.0 else b[i]
    """
    mask = condition.to(a.dtype)
    return a * mask + b * (1.0 - mask)


def ct_equal(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Constant-time equality comparison. Returns 1.0 if equal, 0.0 otherwise.

    No short-circuit evaluation: always computes the full difference.
    For multi-element tensors, compares element-wise.

    The threshold of 0.5 accommodates minor floating-point drift while
    correctly distinguishing integer byte values (which differ by >= 1.0).

    Args:
        a: first operand
        b: second operand

    Returns:
        Tensor of 1.0 (equal) or 0.0 (not equal), same shape as inputs
    """
    diff = (a - b).abs()
    return (diff < 0.5).to(a.dtype)


def ct_less_than(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Constant-time less-than comparison. Returns 1.0 if a < b, else 0.0.

    No short-circuit evaluation or early exit.

    Args:
        a: first operand
        b: second operand

    Returns:
        Tensor of 1.0 (a < b) or 0.0 (a >= b)
    """
    return (a < b).to(a.dtype)


def ct_greater_than(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Constant-time greater-than comparison. Returns 1.0 if a > b, else 0.0.

    Args:
        a: first operand
        b: second operand

    Returns:
        Tensor of 1.0 (a > b) or 0.0 (a <= b)
    """
    return (a > b).to(a.dtype)


def ct_swap(
    condition: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Constant-time conditional swap. Swaps a and b if condition is true.

    Both possible outputs are always computed. This is essential for
    algorithms like Montgomery ladder (used in elliptic curve crypto)
    where the swap pattern must not leak information.

    Args:
        condition: tensor of 0.0 or 1.0 values
        a: first value
        b: second value

    Returns:
        (new_a, new_b) where values are swapped if condition == 1.0
    """
    mask = condition.to(a.dtype)
    new_a = a * (1.0 - mask) + b * mask
    new_b = b * (1.0 - mask) + a * mask
    return new_a, new_b


def ct_byte_lookup(table: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Constant-time table lookup via full linear scan.

    Scans the ENTIRE table for every lookup to avoid leaking the index
    through cache access patterns. On conventional CPUs, ``table[index]``
    loads exactly one cache line, revealing the index to a cache-timing
    attacker. This function always loads ALL entries.

    Complexity: O(n) time and memory access regardless of index value.
    This is deliberately slower than direct indexing for security.

    Args:
        table: 1-D tensor of table entries (e.g., 256 entries for AES S-box)
        index: scalar tensor (float representing an integer index)

    Returns:
        The table entry at the given index
    """
    # Build a mask that is 1.0 only at the target index, 0.0 elsewhere
    indices = torch.arange(table.shape[0], dtype=table.dtype, device=table.device)
    # ct_equal checks |index - i| < 0.5 for each i, so exactly one matches
    mask = ct_equal(indices, index)
    # Dot product selects the matching entry (reads all entries regardless)
    return (table * mask).sum()


def ct_byte_lookup_batch(table: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Constant-time table lookup for a batch of indices.

    Each index in the batch triggers a full table scan. This is the
    vectorized version of ct_byte_lookup for processing multiple bytes.

    Args:
        table: 1-D tensor of table entries [table_size]
        indices: 1-D tensor of index values [batch_size]

    Returns:
        1-D tensor of looked-up values [batch_size]
    """
    table_indices = torch.arange(
        table.shape[0], dtype=table.dtype, device=table.device
    )
    # [batch_size, table_size]: mask is 1.0 where indices[i] matches table index j
    mask = ct_equal(
        indices.unsqueeze(1),
        table_indices.unsqueeze(0),
    )
    # [batch_size]: sum over table dimension selects matching entry
    return (mask * table.unsqueeze(0)).sum(dim=1)


def ct_memcmp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Constant-time memory comparison.

    Compares ALL bytes/elements regardless of where the first mismatch
    occurs. Returns 1.0 if all elements are equal, 0.0 otherwise.

    On conventional CPUs, ``memcmp`` typically returns early on the first
    mismatch, leaking the position of the first differing byte. This
    function always processes every element.

    Args:
        a: first buffer (1-D tensor)
        b: second buffer (1-D tensor)

    Returns:
        Scalar tensor: 1.0 if equal, 0.0 if any element differs
    """
    # Compute per-element equality (always processes all elements)
    element_equal = ct_equal(a, b)
    # All must be equal: product of all element equalities
    # (if any is 0.0, the product is 0.0)
    all_equal = element_equal.prod()
    return ct_equal(all_equal, torch.tensor(1.0, dtype=a.dtype, device=a.device))


# ---------------------------------------------------------------------------
# Additional constant-time arithmetic primitives
# ---------------------------------------------------------------------------

def ct_min(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Constant-time minimum. Returns min(a, b) without branching.

    Args:
        a: first operand
        b: second operand

    Returns:
        Element-wise minimum
    """
    cond = ct_less_than(a, b)
    return ct_select(cond, a, b)


def ct_max(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Constant-time maximum. Returns max(a, b) without branching.

    Args:
        a: first operand
        b: second operand

    Returns:
        Element-wise maximum
    """
    cond = ct_greater_than(a, b)
    return ct_select(cond, a, b)


def ct_abs(a: torch.Tensor) -> torch.Tensor:
    """Constant-time absolute value. No branching on sign bit.

    Args:
        a: input tensor

    Returns:
        |a| computed without data-dependent branches
    """
    is_negative = ct_less_than(a, torch.zeros_like(a))
    return ct_select(is_negative, -a, a)


def ct_clamp(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Constant-time clamp to [lo, hi] range without branching.

    Args:
        x: input tensor
        lo: lower bound
        hi: upper bound

    Returns:
        x clamped to [lo, hi]
    """
    return ct_max(lo, ct_min(x, hi))


def ct_rotate_left(x: torch.Tensor, n: int, bits: int = 8) -> torch.Tensor:
    """Constant-time left rotation of byte values.

    Rotates bit positions left by n within a `bits`-wide word.
    "Left rotation" means toward higher bit significance:
    bit i moves to position (i + n) mod bits.

    Always decomposes into all bits regardless of input.

    Args:
        x: float tensor of byte values [0, 255]
        n: number of positions to rotate left
        bits: word width (default 8)

    Returns:
        Rotated byte values
    """
    n = n % bits
    if n == 0:
        return x.clone()
    bit_vec = _to_bits(x, n_bits=bits)
    # _to_bits returns LSB-first: bit_vec[..., 0] is bit 0 (weight 1),
    # bit_vec[..., 7] is bit 7 (weight 128).
    # Rotate left by n = each bit at position i moves to position (i+n) % bits.
    # In LSB-first layout, this means we roll the array to the RIGHT by n:
    # the top n bits wrap around to the bottom.
    rotated = torch.cat([bit_vec[..., -n:], bit_vec[..., :-n]], dim=-1)
    return _from_bits(rotated)


def ct_rotate_right(x: torch.Tensor, n: int, bits: int = 8) -> torch.Tensor:
    """Constant-time right rotation of byte values.

    Rotates bit positions right by n within a `bits`-wide word.

    Args:
        x: float tensor of byte values [0, 255]
        n: number of positions to rotate right
        bits: word width (default 8)

    Returns:
        Rotated byte values
    """
    n = n % bits
    if n == 0:
        return x.clone()
    # Right rotation by n = left rotation by (bits - n)
    return ct_rotate_left(x, bits - n, bits)


def ct_is_zero(x: torch.Tensor) -> torch.Tensor:
    """Constant-time zero test. Returns 1.0 if x == 0, else 0.0.

    Args:
        x: input tensor

    Returns:
        Tensor of 0.0/1.0 values
    """
    return ct_equal(x, torch.zeros_like(x))
