"""CPURegistry: Verified CPU primitives for KVRM-CPU.

This module implements the registry pattern for CPU operations,
where each operation is a verified, frozen primitive that transforms
state in a predictable, auditable way.

Registry Keys:
    OP_MOV_REG_IMM: Load immediate value into register
    OP_MOV_REG_REG: Copy register to register
    OP_ADD: Add two registers, store result
    OP_SUB: Subtract two registers, store result
    OP_MUL: Multiply two registers, store result
    OP_INC: Increment register by 1
    OP_DEC: Decrement register by 1
    OP_CMP: Compare two registers, set flags
    OP_JMP: Unconditional jump to address
    OP_JZ: Jump if zero flag set
    OP_JNZ: Jump if zero flag not set
    OP_JS: Jump if sign flag set (negative)
    OP_JNS: Jump if sign flag not set (non-negative)
    OP_HALT: Stop execution
    OP_NOP: No operation
    OP_INVALID: Error handling for invalid instructions

Each primitive is a pure function: (CPUState, params) -> CPUState
All state mutations are immutable, returning new state objects.
"""

from typing import Dict, Callable, Any, Optional
from copy import deepcopy
from .state import CPUState, INT32_MIN, INT32_MAX


class CPURegistry:
    """Verified registry of CPU primitives.

    The registry is frozen after initialization to ensure
    no runtime modifications can occur.

    Attributes:
        _primitives: Dictionary mapping operation keys to handler functions
        _frozen: Whether the registry is locked against modifications
    """

    def __init__(self):
        """Initialize registry with all CPU primitives."""
        self._primitives: Dict[str, Callable[[CPUState, Dict[str, Any]], CPUState]] = {}
        self._frozen = False
        self._register_all_primitives()
        self.freeze()

    def _register_all_primitives(self) -> None:
        """Register all CPU operation primitives."""
        # Data movement
        self.register("OP_MOV_REG_IMM", self._op_mov_reg_imm)
        self.register("OP_MOV_REG_REG", self._op_mov_reg_reg)

        # Arithmetic
        self.register("OP_ADD", self._op_add)
        self.register("OP_SUB", self._op_sub)
        self.register("OP_MUL", self._op_mul)
        self.register("OP_INC", self._op_inc)
        self.register("OP_DEC", self._op_dec)

        # Comparison
        self.register("OP_CMP", self._op_cmp)

        # Control flow
        self.register("OP_JMP", self._op_jmp)
        self.register("OP_JZ", self._op_jz)
        self.register("OP_JNZ", self._op_jnz)
        self.register("OP_JS", self._op_js)
        self.register("OP_JNS", self._op_jns)

        # Special
        self.register("OP_HALT", self._op_halt)
        self.register("OP_NOP", self._op_nop)
        self.register("OP_INVALID", self._op_invalid)

    def register(self, key: str, handler: Callable[[CPUState, Dict[str, Any]], CPUState]) -> None:
        """Register a primitive operation.

        Args:
            key: Operation key (e.g., "OP_ADD")
            handler: Function that takes (state, params) and returns new state

        Raises:
            RuntimeError: If registry is frozen
            ValueError: If key already registered
        """
        if self._frozen:
            raise RuntimeError("Cannot register primitives: registry is frozen")
        if key in self._primitives:
            raise ValueError(f"Primitive already registered: {key}")
        self._primitives[key] = handler

    def freeze(self) -> None:
        """Freeze the registry to prevent further modifications."""
        self._frozen = True

    def is_frozen(self) -> bool:
        """Check if registry is frozen."""
        return self._frozen

    def get_valid_keys(self) -> set:
        """Get set of all valid operation keys."""
        return set(self._primitives.keys())

    def execute(self, state: CPUState, key: str, params: Dict[str, Any]) -> CPUState:
        """Execute a registered primitive.

        Args:
            state: Current CPU state
            key: Operation key
            params: Operation parameters

        Returns:
            New CPU state after execution

        Raises:
            KeyError: If key not in registry
        """
        if key not in self._primitives:
            raise KeyError(f"Unknown operation key: {key}")

        handler = self._primitives[key]
        new_state = handler(state, params)

        # Always increment cycle count after execution
        return new_state.increment_cycle()

    # =========================================================================
    # Data Movement Primitives
    # =========================================================================

    def _op_mov_reg_imm(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        """MOV Rd, imm - Load immediate value into register.

        Params:
            dest: Destination register (R0-R7)
            value: Immediate value (32-bit signed integer)

        Returns:
            New state with register updated
        """
        dest = params["dest"]
        value = params["value"]

        # Set register and update flags based on value
        new_state = state.set_register(dest, value)
        new_state = new_state.set_flags(value)
        return new_state.increment_pc()

    def _op_mov_reg_reg(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        """MOV Rd, Rs - Copy value from source register to destination.

        Params:
            dest: Destination register (R0-R7)
            src: Source register (R0-R7)

        Returns:
            New state with destination register updated
        """
        dest = params["dest"]
        src = params["src"]

        value = state.get_register(src)
        new_state = state.set_register(dest, value)
        new_state = new_state.set_flags(value)
        return new_state.increment_pc()

    # =========================================================================
    # Arithmetic Primitives
    # =========================================================================

    def _op_add(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        """ADD Rd, Rs1, Rs2 - Add two registers, store in destination.

        Params:
            dest: Destination register
            src1: First source register
            src2: Second source register

        Returns:
            New state with result in dest, flags updated
        """
        dest = params["dest"]
        src1 = params["src1"]
        src2 = params["src2"]

        val1 = state.get_register(src1)
        val2 = state.get_register(src2)
        result = self._clamp_value(val1 + val2)

        new_state = state.set_register(dest, result)
        new_state = new_state.set_flags(result)
        return new_state.increment_pc()

    def _op_sub(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        """SUB Rd, Rs1, Rs2 - Subtract Rs2 from Rs1, store in destination.

        Params:
            dest: Destination register
            src1: First source register (minuend)
            src2: Second source register (subtrahend)

        Returns:
            New state with result in dest, flags updated
        """
        dest = params["dest"]
        src1 = params["src1"]
        src2 = params["src2"]

        val1 = state.get_register(src1)
        val2 = state.get_register(src2)
        result = self._clamp_value(val1 - val2)

        new_state = state.set_register(dest, result)
        new_state = new_state.set_flags(result)
        return new_state.increment_pc()

    def _op_mul(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        """MUL Rd, Rs1, Rs2 - Multiply two registers, store in destination.

        Params:
            dest: Destination register
            src1: First source register
            src2: Second source register

        Returns:
            New state with result in dest, flags updated
        """
        dest = params["dest"]
        src1 = params["src1"]
        src2 = params["src2"]

        val1 = state.get_register(src1)
        val2 = state.get_register(src2)
        result = self._clamp_value(val1 * val2)

        new_state = state.set_register(dest, result)
        new_state = new_state.set_flags(result)
        return new_state.increment_pc()

    def _op_inc(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        """INC Rd - Increment register by 1.

        Params:
            dest: Register to increment

        Returns:
            New state with register incremented, flags updated
        """
        dest = params["dest"]
        value = state.get_register(dest)
        result = self._clamp_value(value + 1)

        new_state = state.set_register(dest, result)
        new_state = new_state.set_flags(result)
        return new_state.increment_pc()

    def _op_dec(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        """DEC Rd - Decrement register by 1.

        Params:
            dest: Register to decrement

        Returns:
            New state with register decremented, flags updated
        """
        dest = params["dest"]
        value = state.get_register(dest)
        result = self._clamp_value(value - 1)

        new_state = state.set_register(dest, result)
        new_state = new_state.set_flags(result)
        return new_state.increment_pc()

    # =========================================================================
    # Comparison Primitives
    # =========================================================================

    def _op_cmp(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        """CMP Rs1, Rs2 - Compare two registers, set flags.

        Sets ZF if Rs1 == Rs2 (difference is zero)
        Sets SF if Rs1 < Rs2 (difference is negative)

        Params:
            src1: First source register
            src2: Second source register

        Returns:
            New state with flags updated (no register changes)
        """
        src1 = params["src1"]
        src2 = params["src2"]

        val1 = state.get_register(src1)
        val2 = state.get_register(src2)
        diff = val1 - val2

        # Set flags based on comparison result
        new_state = state.set_flags(diff)
        return new_state.increment_pc()

    # =========================================================================
    # Control Flow Primitives
    # =========================================================================

    def _op_jmp(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        """JMP addr - Unconditional jump to address.

        Params:
            addr: Target address (instruction index)

        Returns:
            New state with PC set to target address
        """
        addr = params["addr"]
        return state.set_pc(addr)

    def _op_jz(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        """JZ addr - Jump if zero flag is set.

        Params:
            addr: Target address (instruction index)

        Returns:
            New state with PC set to target if ZF=1, else PC+1
        """
        addr = params["addr"]

        if state.flags["ZF"]:
            return state.set_pc(addr)
        else:
            return state.increment_pc()

    def _op_jnz(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        """JNZ addr - Jump if zero flag is not set.

        Params:
            addr: Target address (instruction index)

        Returns:
            New state with PC set to target if ZF=0, else PC+1
        """
        addr = params["addr"]

        if not state.flags["ZF"]:
            return state.set_pc(addr)
        else:
            return state.increment_pc()

    def _op_js(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        """JS addr - Jump if sign flag is set (negative result).

        Params:
            addr: Target address (instruction index)

        Returns:
            New state with PC set to target if SF=1, else PC+1
        """
        addr = params["addr"]

        if state.flags["SF"]:
            return state.set_pc(addr)
        else:
            return state.increment_pc()

    def _op_jns(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        """JNS addr - Jump if sign flag is not set (non-negative result).

        Params:
            addr: Target address (instruction index)

        Returns:
            New state with PC set to target if SF=0, else PC+1
        """
        addr = params["addr"]

        if not state.flags["SF"]:
            return state.set_pc(addr)
        else:
            return state.increment_pc()

    # =========================================================================
    # Special Primitives
    # =========================================================================

    def _op_halt(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        """HALT - Stop execution.

        Returns:
            New state with halted flag set
        """
        return state.set_halted(True)

    def _op_nop(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        """NOP - No operation.

        Returns:
            New state with only PC incremented
        """
        return state.increment_pc()

    def _op_invalid(self, state: CPUState, params: Dict[str, Any]) -> CPUState:
        """INVALID - Handle invalid/unrecognized instruction.

        Params:
            raw: Original instruction string (for debugging)

        Returns:
            New state with halted flag set (fail-safe)
        """
        raw = params.get("raw", "UNKNOWN")
        # For safety, halt on invalid instruction
        # In a real CPU this might trigger an exception/interrupt
        return state.set_halted(True)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _clamp_value(self, value: int) -> int:
        """Clamp value to 32-bit signed integer range.

        Args:
            value: Integer value to clamp

        Returns:
            Value clamped to [INT32_MIN, INT32_MAX]
        """
        return max(INT32_MIN, min(INT32_MAX, value))


# Singleton registry instance
_registry: Optional[CPURegistry] = None


def get_registry() -> CPURegistry:
    """Get the singleton CPU registry instance.

    Returns:
        The frozen CPURegistry instance
    """
    global _registry
    if _registry is None:
        _registry = CPURegistry()
    return _registry
