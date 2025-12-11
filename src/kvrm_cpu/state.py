"""CPUState: Immutable state representation for KVRM-CPU.

This module defines the core state structure for the CPU emulator,
following functional programming principles for auditability and tracing.

State Components:
    - Registers: R0-R7 (8 general-purpose 32-bit signed integers)
    - PC: Program counter
    - Flags: ZF (zero), SF (sign/negative)
    - Memory: Program instructions (list of strings)
    - Halted: Execution termination flag
    - Cycle count: Total executed cycles

All state mutations return new state objects, preserving immutability
for full execution trace capability.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from copy import deepcopy


# 32-bit signed integer bounds
INT32_MIN = -(2**31)
INT32_MAX = (2**31) - 1


@dataclass
class CPUState:
    """Immutable CPU state representation.

    Attributes:
        registers: Dictionary mapping register names (R0-R7) to 32-bit signed values
        pc: Program counter (current instruction address)
        flags: Dictionary of CPU flags (ZF=zero, SF=sign/negative)
        memory: List of instruction strings (program loaded in memory)
        halted: Whether the CPU has executed HALT
        cycle_count: Number of execution cycles completed
    """
    registers: Dict[str, int] = field(default_factory=lambda: {
        "R0": 0, "R1": 0, "R2": 0, "R3": 0,
        "R4": 0, "R5": 0, "R6": 0, "R7": 0
    })
    pc: int = 0
    flags: Dict[str, bool] = field(default_factory=lambda: {
        "ZF": False,  # Zero flag
        "SF": False   # Sign flag (negative)
    })
    memory: List[str] = field(default_factory=list)
    halted: bool = False
    cycle_count: int = 0

    def snapshot(self) -> dict:
        """Create an immutable snapshot of current state for tracing.

        Returns:
            Dictionary containing deep copy of all state components
        """
        return {
            "registers": deepcopy(self.registers),
            "pc": self.pc,
            "flags": deepcopy(self.flags),
            "halted": self.halted,
            "cycle_count": self.cycle_count,
            # Note: memory excluded from snapshot for efficiency (it doesn't change)
        }

    def validate(self) -> bool:
        """Validate state integrity.

        Checks:
            - All registers exist and are within 32-bit signed bounds
            - PC is non-negative and within memory bounds (or at end)
            - Flags are boolean

        Returns:
            True if state is valid, False otherwise
        """
        # Check registers
        expected_regs = {"R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"}
        if set(self.registers.keys()) != expected_regs:
            return False

        for reg, value in self.registers.items():
            if not isinstance(value, int):
                return False
            if value < INT32_MIN or value > INT32_MAX:
                return False

        # Check PC
        if self.pc < 0:
            return False
        if self.memory and self.pc > len(self.memory):
            return False

        # Check flags
        if set(self.flags.keys()) != {"ZF", "SF"}:
            return False
        for flag_value in self.flags.values():
            if not isinstance(flag_value, bool):
                return False

        # Check cycle count
        if self.cycle_count < 0:
            return False

        return True

    def get_register(self, reg: str) -> int:
        """Get value of a register.

        Args:
            reg: Register name (R0-R7, case insensitive)

        Returns:
            Register value

        Raises:
            KeyError: If register doesn't exist
        """
        reg_upper = reg.upper()
        if reg_upper not in self.registers:
            raise KeyError(f"Invalid register: {reg}")
        return self.registers[reg_upper]

    def set_register(self, reg: str, value: int) -> "CPUState":
        """Create new state with updated register value.

        Args:
            reg: Register name (R0-R7, case insensitive)
            value: New value (will be clamped to 32-bit signed range)

        Returns:
            New CPUState with updated register

        Raises:
            KeyError: If register doesn't exist
        """
        reg_upper = reg.upper()
        if reg_upper not in self.registers:
            raise KeyError(f"Invalid register: {reg}")

        # Clamp to 32-bit signed range
        clamped_value = max(INT32_MIN, min(INT32_MAX, value))

        new_registers = deepcopy(self.registers)
        new_registers[reg_upper] = clamped_value

        return CPUState(
            registers=new_registers,
            pc=self.pc,
            flags=deepcopy(self.flags),
            memory=self.memory,  # Shared reference (immutable)
            halted=self.halted,
            cycle_count=self.cycle_count
        )

    def set_flags(self, value: int) -> "CPUState":
        """Create new state with flags updated based on a value.

        Args:
            value: Value to derive flags from

        Returns:
            New CPUState with updated flags
        """
        new_flags = {
            "ZF": value == 0,
            "SF": value < 0
        }

        return CPUState(
            registers=deepcopy(self.registers),
            pc=self.pc,
            flags=new_flags,
            memory=self.memory,
            halted=self.halted,
            cycle_count=self.cycle_count
        )

    def increment_pc(self) -> "CPUState":
        """Create new state with PC incremented by 1.

        Returns:
            New CPUState with incremented PC
        """
        return CPUState(
            registers=deepcopy(self.registers),
            pc=self.pc + 1,
            flags=deepcopy(self.flags),
            memory=self.memory,
            halted=self.halted,
            cycle_count=self.cycle_count
        )

    def set_pc(self, new_pc: int) -> "CPUState":
        """Create new state with new PC value.

        Args:
            new_pc: New program counter value

        Returns:
            New CPUState with updated PC
        """
        return CPUState(
            registers=deepcopy(self.registers),
            pc=new_pc,
            flags=deepcopy(self.flags),
            memory=self.memory,
            halted=self.halted,
            cycle_count=self.cycle_count
        )

    def set_halted(self, halted: bool = True) -> "CPUState":
        """Create new state with halted flag set.

        Args:
            halted: Halted state (default True)

        Returns:
            New CPUState with halted flag
        """
        return CPUState(
            registers=deepcopy(self.registers),
            pc=self.pc,
            flags=deepcopy(self.flags),
            memory=self.memory,
            halted=halted,
            cycle_count=self.cycle_count
        )

    def increment_cycle(self) -> "CPUState":
        """Create new state with cycle count incremented.

        Returns:
            New CPUState with incremented cycle count
        """
        return CPUState(
            registers=deepcopy(self.registers),
            pc=self.pc,
            flags=deepcopy(self.flags),
            memory=self.memory,
            halted=self.halted,
            cycle_count=self.cycle_count + 1
        )

    def dump_registers(self) -> Dict[str, int]:
        """Get a copy of all register values.

        Returns:
            Dictionary of register names to values
        """
        return deepcopy(self.registers)

    def __str__(self) -> str:
        """Human-readable state representation."""
        regs = " ".join(f"{k}={v}" for k, v in sorted(self.registers.items()))
        flags = " ".join(f"{k}={int(v)}" for k, v in self.flags.items())
        return f"[Cycle {self.cycle_count}] PC={self.pc} {regs} {flags} {'HALTED' if self.halted else ''}"


def create_initial_state(program: List[str]) -> CPUState:
    """Create initial CPU state with a loaded program.

    Args:
        program: List of instruction strings

    Returns:
        Fresh CPUState with program loaded
    """
    return CPUState(
        registers={"R0": 0, "R1": 0, "R2": 0, "R3": 0, "R4": 0, "R5": 0, "R6": 0, "R7": 0},
        pc=0,
        flags={"ZF": False, "SF": False},
        memory=program,
        halted=False,
        cycle_count=0
    )
