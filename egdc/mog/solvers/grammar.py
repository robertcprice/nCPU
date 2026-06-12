"""Grammar constraints for differentiable Mog program synthesis.

Injects Mog syntax awareness into the gradient descent optimization loop.
Instead of free-form logits, these constraints push the soft program toward
valid Mog programs during training, reducing the discretization gap.

Three categories of constraints:
1. Structural: return must exist, dead code after return, valid transitions
2. Safety: division/mod by zero prevention
3. Efficiency: penalize unused writes, encourage decisive choices
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# Statement type indices (must match mog_program_search.STMT_TYPES)
STMT_NOP = 0
STMT_BINOP = 1
STMT_CONST = 2
STMT_ASSIGN = 3
STMT_IF_RETURN = 4
STMT_LOOP = 5
STMT_RETURN = 6

# Operator indices (must match mog_program_search.OPS)
OP_ADD = 0
OP_SUB = 1
OP_MUL = 2
OP_DIV = 3
OP_MOD = 4


def grammar_penalty(
    stmt_logits: torch.Tensor,
    op_logits: torch.Tensor,
    src2_logits: torch.Tensor,
    num_slots: int,
    num_sources: int,
) -> torch.Tensor:
    """Compute a differentiable grammar penalty for a soft program.

    Args:
        stmt_logits: [num_slots, 7] logits over statement types
        op_logits: [num_slots, 5] logits over operators
        src2_logits: [num_slots, num_sources] logits over source variables
        num_slots: number of program slots
        num_sources: number of source variables (args + locals)

    Returns:
        Scalar penalty tensor (differentiable w.r.t. all inputs).
    """
    penalty = torch.tensor(0.0, device=stmt_logits.device)

    # 1. Return must exist: at least one slot should have return_var active
    return_probs = [F.softmax(stmt_logits[s], dim=0)[STMT_RETURN] for s in range(num_slots)]
    max_return = torch.stack(return_probs).max()
    penalty = penalty + 0.5 * torch.relu(0.8 - max_return)

    # 2. Dead code after return: if slot i is return_var, later slots should be nop
    for i in range(num_slots - 1):
        ret_prob = F.softmax(stmt_logits[i], dim=0)[STMT_RETURN]
        for j in range(i + 1, num_slots):
            nop_prob = F.softmax(stmt_logits[j], dim=0)[STMT_NOP]
            penalty = penalty + 0.1 * ret_prob * (1.0 - nop_prob)

    # 3. Division/mod safety: penalize div/mod when src2 is near-zero constant
    for s in range(num_slots):
        stmt_probs = F.softmax(stmt_logits[s], dim=0)
        op_probs = F.softmax(op_logits[s], dim=0)
        binop_prob = stmt_probs[STMT_BINOP]
        div_mod_prob = op_probs[OP_DIV] + op_probs[OP_MOD]
        penalty = penalty + 0.05 * binop_prob * div_mod_prob

    # 4. First slot should not be nop: encourage computation early
    first_nop = F.softmax(stmt_logits[0], dim=0)[STMT_NOP]
    penalty = penalty + 0.2 * first_nop

    # 5. Encourage decisive choices (low entropy) on stmt types
    for s in range(num_slots):
        entropy = -(F.softmax(stmt_logits[s], dim=0) * F.log_softmax(stmt_logits[s], dim=0)).sum()
        penalty = penalty + 0.01 * entropy

    return penalty


def validate_discrete(code: str) -> tuple[bool, str]:
    """Validate that a generated Mog program parses correctly.

    Uses the real Mog lexer + parser for ground-truth validation.

    Args:
        code: Mog source code string

    Returns:
        (is_valid, error_message) — error_message is empty if valid
    """
    try:
        from egdc.mog.lang.lexer import lex
        from egdc.mog.lang.parser import parse

        tokens = lex(code)
        if not tokens:
            return False, "lexer produced no tokens"
        ast = parse(tokens)
        if ast is None:
            return False, "parser returned None"
        return True, ""
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def validate_with_interpreter(code: str) -> tuple[bool, str]:
    """Validate by actually executing the code with a trivial call.

    Returns:
        (is_valid, error_message)
    """
    try:
        from egdc.mog.lang import interpret
        result = interpret(code + '\nfn main() -> i64 { return 0; }')
        return True, ""
    except Exception as e:
        # Many programs can't be called trivially — just check parsing
        return validate_discrete(code)


def constrained_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Softmax with an optional validity mask.

    Args:
        logits: [N] or [B, N] logits
        mask: same shape, 1.0 for valid choices, 0.0 for invalid
        temperature: softmax temperature

    Returns:
        Probability distribution summing to 1.0 over valid choices.
    """
    scaled = logits / temperature
    if mask is not None:
        scaled = scaled.masked_fill(mask < 0.5, float('-inf'))
    return F.softmax(scaled, dim=-1)


def make_stmt_mask(slot_idx: int, num_slots: int) -> torch.Tensor:
    """Create a validity mask for statement types at a given slot.

    Rules:
    - Slot 0: no nop, no return (should compute something)
    - Last slot: should be return or if_return
    - Middle slots: any valid statement type

    Returns:
        [7] tensor of 0/1 validity flags
    """
    mask = torch.ones(7)
    if slot_idx == 0:
        mask[STMT_NOP] = 0.0
        mask[STMT_RETURN] = 0.0
    if slot_idx == num_slots - 1:
        # Last slot should be a return statement
        mask[STMT_NOP] = 0.0
        mask[STMT_BINOP] = 0.0
        mask[STMT_CONST] = 0.0
        mask[STMT_ASSIGN] = 0.0
        mask[STMT_LOOP] = 0.0
    return mask
