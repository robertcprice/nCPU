"""
MOONSHOT #2: Thermodynamic Program Annealing

Treats program space as a Boltzmann distribution where:
- Energy E(program) measures program quality (lower = better)
- Temperature T controls exploration randomness
- Cooling schedule anneals from hot (random) to cold (optimal)
- Phase transitions reveal emergent computational structures (loops, conditionals)
- Entropy measures novelty; free energy measures surprise

Based on Grok's insight: "Program space as Boltzmann distribution; temperature =
exploration randomness. Cooling schedule: Start hot (random comps) -> anneal to
low-energy (optimal algos). Phase transitions reveal loops/conditionals."

Architecture:
    1. ProgramThermodynamics - Energy function computing E(program)
    2. CoolingSchedule - Temperature evolution strategies
    3. PhaseTransitionDetector - Identifies emergent structures
    4. ThermodynamicAnnealer - Main simulated annealing engine
    5. NoveltyEntropy - Information-theoretic novelty metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable, Set
from enum import Enum, auto
from collections import defaultdict
import hashlib
import sys
from pathlib import Path

# Add parent directory to path for SPNC imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from spnc.instruction_set import (
    Program, Instruction, ARM64Opcode, NUM_REGISTERS,
    make_add, make_sub, make_mul, make_div, make_mov_imm, make_mov_reg,
    make_cmp, make_cmp_imm, make_branch, make_beq, make_bne, make_blt,
    make_bge, make_bgt, make_ble, make_ret, make_nop
)


# =============================================================================
# ENERGY FUNCTION: E(program)
# =============================================================================

class EnergyComponent(Enum):
    """Components of the program energy function."""
    CORRECTNESS = auto()     # How well it satisfies test cases
    LENGTH = auto()          # Program length penalty (Kolmogorov-inspired)
    COMPLEXITY = auto()      # Structural complexity
    SEMANTIC = auto()        # Semantic coherence
    EXECUTION = auto()       # Execution characteristics


@dataclass
class EnergyBreakdown:
    """Detailed breakdown of program energy."""
    total: float
    components: Dict[EnergyComponent, float]
    weights: Dict[EnergyComponent, float]

    def to_dict(self) -> Dict[str, float]:
        return {
            'total': self.total,
            **{f'E_{k.name.lower()}': v for k, v in self.components.items()}
        }


class ProgramThermodynamics:
    """
    Computes the energy E(program) of a program state.

    Energy function design principles:
    1. Lower energy = better program
    2. Correct programs have much lower energy than incorrect ones
    3. Shorter correct programs have lower energy (Occam's razor)
    4. Semantically coherent programs have lower energy
    5. Energy landscape has structure that enables efficient annealing

    E(program) = w_c * E_correctness + w_l * E_length + w_x * E_complexity + w_s * E_semantic
    """

    def __init__(
        self,
        correctness_weight: float = 100.0,  # Correctness dominates
        length_weight: float = 1.0,          # Mild length penalty
        complexity_weight: float = 0.5,      # Structural complexity
        semantic_weight: float = 2.0,        # Semantic coherence
        execution_weight: float = 5.0,       # Execution efficiency
        device: str = 'cpu'
    ):
        self.weights = {
            EnergyComponent.CORRECTNESS: correctness_weight,
            EnergyComponent.LENGTH: length_weight,
            EnergyComponent.COMPLEXITY: complexity_weight,
            EnergyComponent.SEMANTIC: semantic_weight,
            EnergyComponent.EXECUTION: execution_weight,
        }
        self.device = device

        # Cache for expensive computations
        self._energy_cache: Dict[str, EnergyBreakdown] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Semantic patterns for bonus energy reduction
        self.semantic_patterns = self._initialize_semantic_patterns()

    def _initialize_semantic_patterns(self) -> List[Dict[str, Any]]:
        """Define semantically meaningful instruction patterns."""
        return [
            # Double: x = x + x (semantically clean doubling)
            {
                'name': 'double',
                'pattern': lambda instrs: any(
                    i.opcode == ARM64Opcode.ADD and i.rd == i.rn == i.rm
                    for i in instrs
                ),
                'bonus': -5.0  # Energy reduction
            },
            # Increment by 1
            {
                'name': 'increment',
                'pattern': lambda instrs: any(
                    i.opcode == ARM64Opcode.ADD and i.is_immediate and i.rm == 1
                    for i in instrs
                ),
                'bonus': -3.0
            },
            # Clear register (MOV Xn, #0)
            {
                'name': 'clear',
                'pattern': lambda instrs: any(
                    i.opcode == ARM64Opcode.MOV and i.is_immediate and i.rm == 0
                    for i in instrs
                ),
                'bonus': -2.0
            },
            # Conditional loop structure (CMP followed by conditional branch)
            {
                'name': 'conditional_loop',
                'pattern': lambda instrs: any(
                    i < len(instrs) - 1 and
                    instrs[i].opcode == ARM64Opcode.CMP and
                    ARM64Opcode.is_conditional_branch(instrs[i+1].opcode)
                    for i in range(len(instrs))
                ),
                'bonus': -8.0  # Strong reward for loop structure
            },
            # Square pattern: MUL Xn, Xm, Xm where n != m
            {
                'name': 'square',
                'pattern': lambda instrs: any(
                    i.opcode == ARM64Opcode.MUL and i.rn == i.rm
                    for i in instrs
                ),
                'bonus': -10.0  # Strong reward for discovering square
            },
        ]

    def _hash_program(self, program: Program) -> str:
        """Create a hash key for caching."""
        instr_str = '|'.join(
            f"{i.opcode.value},{i.rd},{i.rn},{i.rm},{i.is_immediate},{i.branch_target}"
            for i in program.instructions
        )
        return hashlib.md5(instr_str.encode()).hexdigest()

    def compute_energy(
        self,
        program: Program,
        test_cases: List[Tuple[int, int]],
        executor: Optional[Callable] = None,
        use_cache: bool = True
    ) -> EnergyBreakdown:
        """
        Compute total energy of a program.

        Args:
            program: The program to evaluate
            test_cases: List of (input, expected_output) tuples
            executor: Optional function to execute programs
            use_cache: Whether to use energy cache

        Returns:
            EnergyBreakdown with component-wise energy values
        """
        # Check cache
        cache_key = self._hash_program(program) + str(hash(tuple(test_cases)))
        if use_cache and cache_key in self._energy_cache:
            self._cache_hits += 1
            return self._energy_cache[cache_key]
        self._cache_misses += 1

        components = {}

        # E_correctness: How well it satisfies test cases
        components[EnergyComponent.CORRECTNESS] = self._compute_correctness_energy(
            program, test_cases, executor
        )

        # E_length: Program length (Kolmogorov complexity proxy)
        components[EnergyComponent.LENGTH] = self._compute_length_energy(program)

        # E_complexity: Structural complexity
        components[EnergyComponent.COMPLEXITY] = self._compute_complexity_energy(program)

        # E_semantic: Semantic coherence
        components[EnergyComponent.SEMANTIC] = self._compute_semantic_energy(program)

        # E_execution: Execution characteristics
        if executor is not None:
            components[EnergyComponent.EXECUTION] = self._compute_execution_energy(
                program, test_cases, executor
            )
        else:
            components[EnergyComponent.EXECUTION] = 0.0

        # Compute weighted total
        total = sum(
            self.weights[comp] * energy
            for comp, energy in components.items()
        )

        breakdown = EnergyBreakdown(
            total=total,
            components=components,
            weights=self.weights
        )

        # Cache result
        if use_cache:
            self._energy_cache[cache_key] = breakdown

        return breakdown

    def _compute_correctness_energy(
        self,
        program: Program,
        test_cases: List[Tuple[int, int]],
        executor: Optional[Callable]
    ) -> float:
        """
        Compute correctness energy.

        Perfect correctness = 0 energy
        Complete incorrectness = 1.0 energy per test case
        Partial correctness = scaled by distance to correct answer
        """
        if not test_cases:
            return 0.0

        if executor is None:
            # Use simulated execution
            executor = self._simulated_execute

        total_error = 0.0
        for input_val, expected in test_cases:
            try:
                actual = executor(program, input_val)
                if actual == expected:
                    error = 0.0
                else:
                    # Logarithmic error scaling (handles large number differences)
                    diff = abs(actual - expected)
                    error = min(1.0, math.log1p(diff) / 10.0)  # Cap at 1.0
            except Exception:
                error = 1.0  # Maximum error for failed execution
            total_error += error

        # Normalize by number of test cases
        return total_error / len(test_cases)

    def _compute_length_energy(self, program: Program) -> float:
        """
        Compute length energy (Kolmogorov complexity proxy).

        Shorter programs = lower energy (Occam's razor)
        This encodes the MDL (Minimum Description Length) principle.
        """
        n = len(program.instructions)
        # Logarithmic scaling - marginal cost decreases for longer programs
        return math.log1p(n) / 5.0

    def _compute_complexity_energy(self, program: Program) -> float:
        """
        Compute structural complexity energy.

        Measures:
        - Branch complexity (number of branches)
        - Loop nesting
        - Register usage entropy
        """
        instructions = program.instructions

        # Branch complexity
        num_branches = sum(1 for i in instructions if ARM64Opcode.is_branch(i.opcode))
        branch_energy = num_branches * 0.1

        # Register usage diversity (entropy-like)
        reg_usage = defaultdict(int)
        for instr in instructions:
            if instr.opcode not in [ARM64Opcode.RET, ARM64Opcode.NOP]:
                reg_usage[instr.rd] += 1
                reg_usage[instr.rn] += 1
                if not instr.is_immediate:
                    reg_usage[instr.rm] += 1

        if reg_usage:
            total = sum(reg_usage.values())
            probs = [c / total for c in reg_usage.values()]
            reg_entropy = -sum(p * math.log(p + 1e-10) for p in probs)
        else:
            reg_entropy = 0.0

        # High entropy = many registers = higher complexity
        reg_energy = reg_entropy / 3.0

        return branch_energy + reg_energy

    def _compute_semantic_energy(self, program: Program) -> float:
        """
        Compute semantic coherence energy.

        Lower energy for:
        - Recognized computational patterns
        - Meaningful instruction sequences
        - Well-structured control flow
        """
        instructions = program.instructions
        base_energy = 0.5  # Neutral baseline

        # Check for semantic patterns
        pattern_bonus = 0.0
        for pattern in self.semantic_patterns:
            try:
                if pattern['pattern'](instructions):
                    pattern_bonus += pattern['bonus']
            except Exception:
                pass

        # Dead code detection (increases energy)
        dead_code_penalty = self._detect_dead_code(program) * 0.2

        return max(0.0, base_energy + pattern_bonus + dead_code_penalty)

    def _compute_execution_energy(
        self,
        program: Program,
        test_cases: List[Tuple[int, int]],
        executor: Callable
    ) -> float:
        """
        Compute execution efficiency energy.

        Lower energy for:
        - Fast execution
        - Deterministic behavior
        - No infinite loops
        """
        if not test_cases:
            return 0.0

        # Measure execution time
        start = time.perf_counter()
        for input_val, _ in test_cases[:3]:  # Sample for speed
            try:
                executor(program, input_val)
            except Exception:
                return 1.0  # Penalize errors
        elapsed = time.perf_counter() - start

        # Logarithmic time penalty
        return min(1.0, math.log1p(elapsed * 1000) / 5.0)

    def _detect_dead_code(self, program: Program) -> int:
        """Count unreachable instructions."""
        instructions = program.instructions
        n = len(instructions)
        reachable = set([0])  # Start from first instruction

        # Simple reachability analysis
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if i in reachable:
                    instr = instructions[i]
                    if instr.opcode == ARM64Opcode.RET:
                        continue
                    elif instr.opcode == ARM64Opcode.B:
                        target = instr.branch_target
                        if 0 <= target < n and target not in reachable:
                            reachable.add(target)
                            changed = True
                    elif ARM64Opcode.is_conditional_branch(instr.opcode):
                        # Both target and fallthrough
                        target = instr.branch_target
                        if 0 <= target < n and target not in reachable:
                            reachable.add(target)
                            changed = True
                        if i + 1 < n and i + 1 not in reachable:
                            reachable.add(i + 1)
                            changed = True
                    else:
                        if i + 1 < n and i + 1 not in reachable:
                            reachable.add(i + 1)
                            changed = True

        return n - len(reachable)

    def _simulated_execute(self, program: Program, input_val: int) -> int:
        """Simple simulated execution for testing."""
        registers = [0] * 32
        for reg in program.input_registers:
            registers[reg] = input_val

        pc = 0
        max_iterations = 1000
        iterations = 0
        flags = {'N': 0, 'Z': 0, 'C': 0, 'V': 0}

        while pc < len(program.instructions) and iterations < max_iterations:
            instr = program.instructions[pc]
            iterations += 1

            if instr.opcode == ARM64Opcode.RET:
                break
            elif instr.opcode == ARM64Opcode.NOP:
                pc += 1
            elif instr.opcode == ARM64Opcode.ADD:
                b = instr.rm if instr.is_immediate else registers[min(instr.rm, 31)]
                registers[instr.rd] = (registers[instr.rn] + b) & ((1 << 64) - 1)
                pc += 1
            elif instr.opcode == ARM64Opcode.SUB:
                b = instr.rm if instr.is_immediate else registers[min(instr.rm, 31)]
                registers[instr.rd] = (registers[instr.rn] - b) & ((1 << 64) - 1)
                pc += 1
            elif instr.opcode == ARM64Opcode.MUL:
                b = instr.rm if instr.is_immediate else registers[min(instr.rm, 31)]
                registers[instr.rd] = (registers[instr.rn] * b) & ((1 << 64) - 1)
                pc += 1
            elif instr.opcode == ARM64Opcode.MOV:
                if instr.is_immediate:
                    registers[instr.rd] = instr.rm
                else:
                    registers[instr.rd] = registers[min(instr.rm, 31)]
                pc += 1
            elif instr.opcode == ARM64Opcode.CMP:
                a = registers[instr.rn]
                b = instr.rm if instr.is_immediate else registers[min(instr.rm, 31)]
                diff = a - b
                flags['N'] = 1 if diff < 0 else 0
                flags['Z'] = 1 if diff == 0 else 0
                flags['C'] = 1 if a >= b else 0
                pc += 1
            elif instr.opcode == ARM64Opcode.B:
                pc = instr.branch_target
            elif instr.opcode == ARM64Opcode.BEQ:
                pc = instr.branch_target if flags['Z'] else pc + 1
            elif instr.opcode == ARM64Opcode.BNE:
                pc = instr.branch_target if not flags['Z'] else pc + 1
            elif instr.opcode == ARM64Opcode.BLT:
                pc = instr.branch_target if flags['N'] else pc + 1
            elif instr.opcode == ARM64Opcode.BGE:
                pc = instr.branch_target if not flags['N'] else pc + 1
            elif instr.opcode == ARM64Opcode.BGT:
                pc = instr.branch_target if not flags['Z'] and not flags['N'] else pc + 1
            elif instr.opcode == ARM64Opcode.BLE:
                pc = instr.branch_target if flags['Z'] or flags['N'] else pc + 1
            else:
                pc += 1

        return registers[program.output_register]

    def get_cache_stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'size': len(self._energy_cache)
        }

    def clear_cache(self):
        """Clear the energy cache."""
        self._energy_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


# =============================================================================
# COOLING SCHEDULES
# =============================================================================

class CoolingScheduleType(Enum):
    """Types of cooling schedules."""
    LINEAR = auto()
    EXPONENTIAL = auto()
    LOGARITHMIC = auto()
    ADAPTIVE = auto()
    GEOMETRIC = auto()
    REHEAT = auto()  # Allows temperature increases


@dataclass
class CoolingSchedule:
    """
    Temperature evolution over time.

    Temperature controls the probability of accepting worse solutions:
    P(accept) = exp(-delta_E / T)

    High T: Accept almost anything (exploration)
    Low T: Only accept improvements (exploitation)
    """
    schedule_type: CoolingScheduleType
    initial_temperature: float = 100.0
    final_temperature: float = 0.01
    cooling_rate: float = 0.95  # For exponential/geometric
    current_step: int = 0
    max_steps: int = 1000

    # Adaptive schedule parameters
    acceptance_target: float = 0.5  # Target acceptance rate
    adaptation_rate: float = 0.1    # How fast to adapt

    # Reheat parameters
    reheat_threshold: float = 0.1   # When to reheat
    reheat_factor: float = 2.0      # How much to reheat

    def get_temperature(self, step: Optional[int] = None) -> float:
        """Get temperature at current or specified step."""
        step = step if step is not None else self.current_step
        progress = step / max(1, self.max_steps)

        if self.schedule_type == CoolingScheduleType.LINEAR:
            T = self.initial_temperature * (1 - progress) + self.final_temperature * progress

        elif self.schedule_type == CoolingScheduleType.EXPONENTIAL:
            T = self.initial_temperature * (self.cooling_rate ** step)
            T = max(T, self.final_temperature)

        elif self.schedule_type == CoolingScheduleType.LOGARITHMIC:
            # Slower cooling: T(k) = T0 / (1 + alpha * log(1 + k))
            alpha = (self.initial_temperature / self.final_temperature - 1) / math.log1p(self.max_steps)
            T = self.initial_temperature / (1 + alpha * math.log1p(step))

        elif self.schedule_type == CoolingScheduleType.GEOMETRIC:
            T = self.initial_temperature * (self.final_temperature / self.initial_temperature) ** progress

        else:  # ADAPTIVE or REHEAT handled externally
            T = self.initial_temperature * (1 - progress) + self.final_temperature * progress

        return max(T, self.final_temperature)

    def step(self):
        """Advance to next step."""
        self.current_step += 1

    def adapt_temperature(self, acceptance_rate: float, current_temp: float) -> float:
        """
        Adapt temperature based on acceptance rate (for ADAPTIVE schedule).

        If acceptance rate is too low, increase temperature.
        If acceptance rate is too high, decrease temperature.
        """
        if self.schedule_type != CoolingScheduleType.ADAPTIVE:
            return current_temp

        error = acceptance_rate - self.acceptance_target
        adjustment = 1.0 + error * self.adaptation_rate
        new_temp = current_temp * adjustment

        # Keep within bounds
        return max(self.final_temperature, min(self.initial_temperature, new_temp))

    def should_reheat(self, energy_history: List[float]) -> bool:
        """Check if we should reheat (for REHEAT schedule)."""
        if self.schedule_type != CoolingScheduleType.REHEAT:
            return False

        if len(energy_history) < 100:
            return False

        # Check if stuck (no improvement in recent history)
        recent = energy_history[-100:]
        if max(recent) - min(recent) < self.reheat_threshold:
            return True
        return False

    def reheat(self, current_temp: float) -> float:
        """Increase temperature for reheat schedule."""
        return min(self.initial_temperature, current_temp * self.reheat_factor)

    def reset(self):
        """Reset schedule to initial state."""
        self.current_step = 0


# =============================================================================
# PHASE TRANSITION DETECTION
# =============================================================================

class StructuralPhase(Enum):
    """Computational structure phases."""
    RANDOM = auto()          # No meaningful structure
    LINEAR = auto()          # Linear code, no branches
    CONDITIONAL = auto()     # Has conditionals (if/else)
    LOOP = auto()            # Has loops
    RECURSIVE_LIKE = auto()  # Has nested loops or loop patterns
    ALGORITHMIC = auto()     # Complex algorithm detected


@dataclass
class PhaseTransition:
    """Record of a phase transition event."""
    step: int
    temperature: float
    from_phase: StructuralPhase
    to_phase: StructuralPhase
    energy_before: float
    energy_after: float
    program_hash: str
    description: str


class PhaseTransitionDetector:
    """
    Detects phase transitions in program structure during annealing.

    Phase transitions occur when the program undergoes qualitative
    structural changes:
    - Random code -> Linear computation
    - Linear -> Conditional (if/else)
    - Conditional -> Loop structures
    - Loop -> Complex algorithms

    These transitions are analogous to physical phase transitions
    and often occur at critical temperatures.
    """

    def __init__(self):
        self.transition_history: List[PhaseTransition] = []
        self.phase_history: List[Tuple[int, StructuralPhase]] = []
        self.current_phase = StructuralPhase.RANDOM

        # Critical temperature tracking
        self.critical_temps: Dict[str, List[float]] = defaultdict(list)

    def detect_phase(self, program: Program) -> StructuralPhase:
        """Classify the structural phase of a program."""
        instructions = program.instructions

        # Count structural elements
        has_conditional = False
        has_loop = False
        has_nested = False
        branch_targets = set()

        for i, instr in enumerate(instructions):
            if ARM64Opcode.is_conditional_branch(instr.opcode):
                has_conditional = True
                branch_targets.add(instr.branch_target)
            elif instr.opcode == ARM64Opcode.B:
                # Check for backward jump (loop indicator)
                if instr.branch_target < i:
                    has_loop = True
                branch_targets.add(instr.branch_target)

        # Check for nested structures
        if has_loop and len(branch_targets) > 1:
            has_nested = self._detect_nested_loops(program)

        # Classify phase
        if not has_conditional and not has_loop:
            if self._is_meaningful_linear(program):
                return StructuralPhase.LINEAR
            return StructuralPhase.RANDOM
        elif has_loop and has_nested:
            return StructuralPhase.RECURSIVE_LIKE
        elif has_loop:
            return StructuralPhase.LOOP
        elif has_conditional:
            return StructuralPhase.CONDITIONAL
        else:
            return StructuralPhase.LINEAR

    def _is_meaningful_linear(self, program: Program) -> bool:
        """Check if linear code does something meaningful."""
        # Count non-NOP, non-RET instructions
        meaningful = sum(
            1 for i in program.instructions
            if i.opcode not in [ARM64Opcode.NOP, ARM64Opcode.RET]
        )
        return meaningful >= 2

    def _detect_nested_loops(self, program: Program) -> bool:
        """Detect nested loop structures."""
        instructions = program.instructions

        # Find all backward jumps (loops)
        loops = []
        for i, instr in enumerate(instructions):
            if instr.opcode == ARM64Opcode.B and instr.branch_target < i:
                loops.append((instr.branch_target, i))
            elif ARM64Opcode.is_conditional_branch(instr.opcode) and instr.branch_target < i:
                loops.append((instr.branch_target, i))

        # Check for nesting
        for i, (start1, end1) in enumerate(loops):
            for start2, end2 in loops[i+1:]:
                # Nested if one loop is completely inside another
                if start1 < start2 < end2 < end1:
                    return True
                if start2 < start1 < end1 < end2:
                    return True

        return False

    def check_transition(
        self,
        program: Program,
        step: int,
        temperature: float,
        energy: float
    ) -> Optional[PhaseTransition]:
        """
        Check if a phase transition has occurred.

        Returns PhaseTransition if transition detected, None otherwise.
        """
        new_phase = self.detect_phase(program)

        if new_phase != self.current_phase:
            transition = PhaseTransition(
                step=step,
                temperature=temperature,
                from_phase=self.current_phase,
                to_phase=new_phase,
                energy_before=self.phase_history[-1][1] if self.phase_history else 0.0,
                energy_after=energy,
                program_hash=self._hash_program(program),
                description=f"Transition: {self.current_phase.name} -> {new_phase.name}"
            )

            self.transition_history.append(transition)
            self.phase_history.append((step, new_phase))

            # Record critical temperature
            transition_key = f"{self.current_phase.name}->{new_phase.name}"
            self.critical_temps[transition_key].append(temperature)

            self.current_phase = new_phase
            return transition

        return None

    def _hash_program(self, program: Program) -> str:
        """Create hash of program for identification."""
        instr_str = '|'.join(str(i.opcode.value) for i in program.instructions)
        return hashlib.md5(instr_str.encode()).hexdigest()[:8]

    def get_critical_temperatures(self) -> Dict[str, float]:
        """Get average critical temperatures for each transition type."""
        return {
            k: np.mean(v) if v else 0.0
            for k, v in self.critical_temps.items()
        }

    def get_transition_summary(self) -> Dict[str, Any]:
        """Get summary of all phase transitions."""
        return {
            'total_transitions': len(self.transition_history),
            'current_phase': self.current_phase.name,
            'critical_temperatures': self.get_critical_temperatures(),
            'transitions': [
                {
                    'step': t.step,
                    'temp': t.temperature,
                    'from': t.from_phase.name,
                    'to': t.to_phase.name
                }
                for t in self.transition_history
            ]
        }


# =============================================================================
# NOVELTY / ENTROPY METRICS
# =============================================================================

class NoveltyEntropy:
    """
    Information-theoretic metrics for measuring program novelty and surprise.

    Based on:
    - Shannon entropy of program space
    - Kolmogorov complexity approximation
    - Behavioral novelty (output diversity)
    - Structural novelty (instruction patterns)
    """

    def __init__(self):
        # Archive of seen programs for novelty comparison
        self.archive: Dict[str, Program] = {}
        self.behavior_archive: Dict[str, List[int]] = {}  # hash -> outputs

        # Statistics
        self.total_programs_seen = 0
        self.unique_behaviors = 0

    def compute_entropy(self, programs: List[Program]) -> float:
        """
        Compute Shannon entropy of a program population.

        Higher entropy = more diverse population = better exploration.
        """
        if not programs:
            return 0.0

        # Count instruction patterns
        pattern_counts = defaultdict(int)
        for prog in programs:
            for instr in prog.instructions:
                pattern = (instr.opcode, instr.rd, instr.rn)
                pattern_counts[pattern] += 1

        total = sum(pattern_counts.values())
        if total == 0:
            return 0.0

        # Shannon entropy
        entropy = 0.0
        for count in pattern_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def compute_free_energy(
        self,
        program: Program,
        energy: float,
        temperature: float
    ) -> float:
        """
        Compute thermodynamic free energy F = E - T*S.

        Free energy balances:
        - Energy minimization (finding optimal solutions)
        - Entropy maximization (exploring diverse solutions)

        Lower free energy = better trade-off between quality and novelty.
        """
        # Approximate entropy from program structure
        structural_entropy = self._structural_entropy(program)

        # Free energy formula
        free_energy = energy - temperature * structural_entropy

        return free_energy

    def _structural_entropy(self, program: Program) -> float:
        """Compute structural entropy of a single program."""
        instructions = program.instructions
        if not instructions:
            return 0.0

        # Opcode distribution
        opcode_counts = defaultdict(int)
        for instr in instructions:
            opcode_counts[instr.opcode] += 1

        total = sum(opcode_counts.values())
        entropy = 0.0
        for count in opcode_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def compute_novelty(
        self,
        program: Program,
        test_inputs: List[int],
        executor: Callable,
        k: int = 15
    ) -> float:
        """
        Compute behavioral novelty of a program.

        Novelty is the average distance to k-nearest neighbors in behavior space.
        Higher novelty = more unique behavior = more interesting.
        """
        # Compute behavior signature (outputs for test inputs)
        behavior = []
        for inp in test_inputs:
            try:
                out = executor(program, inp)
                behavior.append(out)
            except Exception:
                behavior.append(-1)

        # Add to archive
        prog_hash = self._hash_program(program)
        self.archive[prog_hash] = program
        self.behavior_archive[prog_hash] = behavior
        self.total_programs_seen += 1

        # Check for unique behavior
        behavior_tuple = tuple(behavior)
        existing_behaviors = [tuple(b) for b in self.behavior_archive.values()]
        if behavior_tuple not in existing_behaviors[:-1]:  # Exclude self
            self.unique_behaviors += 1

        # Compute distances to all archived behaviors
        if len(self.behavior_archive) <= 1:
            return 1.0  # First program is maximally novel

        distances = []
        for other_hash, other_behavior in self.behavior_archive.items():
            if other_hash != prog_hash:
                dist = self._behavior_distance(behavior, other_behavior)
                distances.append(dist)

        # Average distance to k nearest neighbors
        distances.sort()
        k = min(k, len(distances))
        if k == 0:
            return 1.0

        novelty = sum(distances[:k]) / k

        # Normalize by expected maximum distance
        return min(1.0, novelty / 1000.0)

    def _behavior_distance(self, b1: List[int], b2: List[int]) -> float:
        """Compute distance between two behavior signatures."""
        if len(b1) != len(b2):
            return float('inf')

        # Use Euclidean distance with log scaling for large numbers
        dist = 0.0
        for v1, v2 in zip(b1, b2):
            if v1 == -1 or v2 == -1:
                dist += 100  # Penalty for execution failures
            else:
                diff = abs(v1 - v2)
                dist += math.log1p(diff)

        return dist

    def _hash_program(self, program: Program) -> str:
        """Create hash for program identification."""
        instr_str = '|'.join(
            f"{i.opcode.value},{i.rd},{i.rn},{i.rm}"
            for i in program.instructions
        )
        return hashlib.md5(instr_str.encode()).hexdigest()

    def compute_surprise(
        self,
        program: Program,
        expected_energy: float,
        actual_energy: float
    ) -> float:
        """
        Compute surprise (prediction error) for a program.

        Surprise = |expected_energy - actual_energy|

        High surprise indicates the program behaves unexpectedly,
        which is interesting from an exploration perspective.
        """
        return abs(expected_energy - actual_energy)

    def get_statistics(self) -> Dict[str, Any]:
        """Get novelty tracking statistics."""
        return {
            'total_programs_seen': self.total_programs_seen,
            'unique_behaviors': self.unique_behaviors,
            'archive_size': len(self.archive),
            'novelty_ratio': self.unique_behaviors / max(1, self.total_programs_seen)
        }


# =============================================================================
# MAIN ANNEALING ENGINE
# =============================================================================

@dataclass
class AnnealingResult:
    """Result of thermodynamic annealing."""
    success: bool
    best_program: Optional[Program]
    best_energy: float
    final_temperature: float
    total_steps: int
    acceptance_rate: float
    phase_transitions: List[PhaseTransition]
    energy_history: List[float]
    temperature_history: List[float]
    time_seconds: float
    statistics: Dict[str, Any]


class ThermodynamicAnnealer:
    """
    Main simulated annealing engine for program synthesis.

    Uses thermodynamic principles to search program space:
    1. Start with high temperature (random exploration)
    2. Gradually cool (focus on improvements)
    3. Detect phase transitions (structural emergence)
    4. Use free energy to balance exploration/exploitation
    """

    def __init__(
        self,
        thermodynamics: Optional[ProgramThermodynamics] = None,
        cooling_schedule: Optional[CoolingSchedule] = None,
        device: str = 'cpu'
    ):
        self.thermodynamics = thermodynamics or ProgramThermodynamics(device=device)
        self.cooling_schedule = cooling_schedule or CoolingSchedule(
            schedule_type=CoolingScheduleType.EXPONENTIAL,
            initial_temperature=100.0,
            final_temperature=0.01,
            cooling_rate=0.995,
            max_steps=10000
        )
        self.device = device

        # Components
        self.phase_detector = PhaseTransitionDetector()
        self.novelty_tracker = NoveltyEntropy()

        # State
        self.current_program: Optional[Program] = None
        self.current_energy: float = float('inf')
        self.best_program: Optional[Program] = None
        self.best_energy: float = float('inf')

        # History
        self.energy_history: List[float] = []
        self.temperature_history: List[float] = []
        self.acceptance_history: List[bool] = []

        # Statistics
        self.total_steps = 0
        self.accepted_steps = 0
        self.rejected_steps = 0
        self.phase_transition_steps: List[int] = []

    def anneal(
        self,
        test_cases: List[Tuple[int, int]],
        input_registers: List[int] = [0],
        output_register: int = 0,
        max_program_length: int = 15,
        initial_program: Optional[Program] = None,
        executor: Optional[Callable] = None,
        verbose: bool = True
    ) -> AnnealingResult:
        """
        Perform simulated annealing to find optimal program.

        Args:
            test_cases: List of (input, expected_output) tuples
            input_registers: Input register indices
            output_register: Output register index
            max_program_length: Maximum program length
            initial_program: Starting program (or random if None)
            executor: Function to execute programs (or use internal)
            verbose: Print progress

        Returns:
            AnnealingResult with best program and statistics
        """
        start_time = time.time()

        if verbose:
            print("=" * 60)
            print("THERMODYNAMIC PROGRAM ANNEALING")
            print("=" * 60)
            print(f"Test cases: {len(test_cases)}")
            print(f"Max program length: {max_program_length}")
            print(f"Schedule: {self.cooling_schedule.schedule_type.name}")
            print(f"T_initial: {self.cooling_schedule.initial_temperature}")
            print(f"T_final: {self.cooling_schedule.final_temperature}")
            print()

        # Initialize
        if initial_program is None:
            self.current_program = self._random_program(
                max_program_length, input_registers, output_register
            )
        else:
            self.current_program = initial_program

        # Compute initial energy
        energy_breakdown = self.thermodynamics.compute_energy(
            self.current_program, test_cases, executor
        )
        self.current_energy = energy_breakdown.total
        self.best_program = self.current_program
        self.best_energy = self.current_energy

        self.energy_history = [self.current_energy]
        self.temperature_history = [self.cooling_schedule.initial_temperature]

        if verbose:
            print(f"Initial energy: {self.current_energy:.4f}")
            print()

        # Main annealing loop
        self.cooling_schedule.reset()

        while self.cooling_schedule.current_step < self.cooling_schedule.max_steps:
            step = self.cooling_schedule.current_step
            temperature = self.cooling_schedule.get_temperature()

            # Generate neighbor
            neighbor = self._generate_neighbor(
                self.current_program,
                max_program_length,
                input_registers,
                output_register,
                temperature
            )

            # Compute neighbor energy
            neighbor_breakdown = self.thermodynamics.compute_energy(
                neighbor, test_cases, executor
            )
            neighbor_energy = neighbor_breakdown.total

            # Metropolis acceptance criterion
            delta_E = neighbor_energy - self.current_energy

            if delta_E < 0:
                # Always accept improvements
                accept = True
            else:
                # Accept with probability exp(-delta_E / T)
                accept_prob = math.exp(-delta_E / max(temperature, 1e-10))
                accept = random.random() < accept_prob

            if accept:
                self.current_program = neighbor
                self.current_energy = neighbor_energy
                self.accepted_steps += 1

                # Update best
                if self.current_energy < self.best_energy:
                    self.best_program = self.current_program
                    self.best_energy = self.current_energy

                    if verbose and step % 100 == 0:
                        print(f"Step {step}: New best energy = {self.best_energy:.4f} (T={temperature:.4f})")
            else:
                self.rejected_steps += 1

            self.acceptance_history.append(accept)

            # Check for phase transitions
            transition = self.phase_detector.check_transition(
                self.current_program, step, temperature, self.current_energy
            )
            if transition:
                self.phase_transition_steps.append(step)
                if verbose:
                    print(f"  PHASE TRANSITION at step {step}: {transition.description}")

            # Record history
            self.energy_history.append(self.current_energy)
            self.temperature_history.append(temperature)

            # Adaptive temperature adjustment
            if self.cooling_schedule.schedule_type == CoolingScheduleType.ADAPTIVE:
                recent_accepts = self.acceptance_history[-100:] if len(self.acceptance_history) >= 100 else self.acceptance_history
                acceptance_rate = sum(recent_accepts) / len(recent_accepts) if recent_accepts else 0.5
                temperature = self.cooling_schedule.adapt_temperature(acceptance_rate, temperature)

            # Check for reheat
            if self.cooling_schedule.should_reheat(self.energy_history):
                new_temp = self.cooling_schedule.reheat(temperature)
                if verbose:
                    print(f"  REHEAT at step {step}: T = {temperature:.4f} -> {new_temp:.4f}")

            # Advance schedule
            self.cooling_schedule.step()
            self.total_steps += 1

            # Check for convergence (perfect program found)
            if self.best_energy < 0.001:  # Near-zero energy = perfect
                if verbose:
                    print(f"\nConverged! Found perfect program at step {step}")
                break

            # Progress reporting
            if verbose and step > 0 and step % 1000 == 0:
                acceptance_rate = self.accepted_steps / max(1, self.total_steps)
                print(f"Step {step}: E={self.current_energy:.4f}, best={self.best_energy:.4f}, "
                      f"T={temperature:.4f}, accept={acceptance_rate:.2%}")

        # Final statistics
        elapsed = time.time() - start_time
        acceptance_rate = self.accepted_steps / max(1, self.total_steps)

        if verbose:
            print()
            print("=" * 60)
            print("ANNEALING COMPLETE")
            print("=" * 60)
            print(f"Total steps: {self.total_steps}")
            print(f"Acceptance rate: {acceptance_rate:.2%}")
            print(f"Best energy: {self.best_energy:.4f}")
            print(f"Phase transitions: {len(self.phase_detector.transition_history)}")
            print(f"Time: {elapsed:.2f}s")
            if self.best_program:
                print(f"\nBest program:")
                print(self.best_program)

        return AnnealingResult(
            success=self.best_energy < 1.0,
            best_program=self.best_program,
            best_energy=self.best_energy,
            final_temperature=self.temperature_history[-1] if self.temperature_history else 0.0,
            total_steps=self.total_steps,
            acceptance_rate=acceptance_rate,
            phase_transitions=self.phase_detector.transition_history,
            energy_history=self.energy_history,
            temperature_history=self.temperature_history,
            time_seconds=elapsed,
            statistics={
                'cache_stats': self.thermodynamics.get_cache_stats(),
                'phase_summary': self.phase_detector.get_transition_summary(),
                'novelty_stats': self.novelty_tracker.get_statistics(),
                'accepted_steps': self.accepted_steps,
                'rejected_steps': self.rejected_steps,
            }
        )

    def _random_program(
        self,
        max_length: int,
        input_registers: List[int],
        output_register: int
    ) -> Program:
        """Generate a random program."""
        length = random.randint(2, max_length)
        instructions = []

        for _ in range(length - 1):
            instructions.append(self._random_instruction())

        instructions.append(make_ret())

        return Program(
            instructions=instructions,
            input_registers=input_registers,
            output_register=output_register
        )

    def _random_instruction(self) -> Instruction:
        """Generate a random instruction."""
        opcode = random.choice([
            ARM64Opcode.ADD, ARM64Opcode.SUB, ARM64Opcode.MUL,
            ARM64Opcode.AND, ARM64Opcode.ORR, ARM64Opcode.EOR,
            ARM64Opcode.MOV, ARM64Opcode.CMP,
            ARM64Opcode.LSL, ARM64Opcode.LSR,
        ])

        rd = random.randint(0, 10)
        rn = random.randint(0, 10)
        rm = random.randint(0, 10)
        is_immediate = random.random() < 0.3

        if is_immediate:
            rm = random.randint(0, 100)

        return Instruction(
            opcode=opcode,
            rd=rd,
            rn=rn,
            rm=rm,
            is_immediate=is_immediate
        )

    def _generate_neighbor(
        self,
        program: Program,
        max_length: int,
        input_registers: List[int],
        output_register: int,
        temperature: float
    ) -> Program:
        """
        Generate a neighbor program by mutation.

        Mutation intensity is temperature-dependent:
        - High T: Large mutations (add/remove instructions)
        - Low T: Small mutations (tweak registers/immediates)
        """
        # Choose mutation type based on temperature
        # Higher temperature = more aggressive mutations
        mutation_probs = self._get_mutation_probabilities(temperature)

        mutation_type = random.choices(
            ['add', 'remove', 'modify', 'swap', 'pattern'],
            weights=mutation_probs
        )[0]

        instructions = list(program.instructions)

        if mutation_type == 'add' and len(instructions) < max_length:
            # Add a random instruction
            pos = random.randint(0, len(instructions) - 1)

            # At high temperature, might add complex structures
            if temperature > 50 and random.random() < 0.3:
                new_instr = self._generate_structural_instruction(len(instructions), pos)
            else:
                new_instr = self._random_instruction()

            instructions.insert(pos, new_instr)

        elif mutation_type == 'remove' and len(instructions) > 2:
            # Remove a random non-RET instruction
            pos = random.randint(0, len(instructions) - 2)
            del instructions[pos]

        elif mutation_type == 'modify':
            # Modify an instruction
            pos = random.randint(0, len(instructions) - 2)
            instructions[pos] = self._mutate_instruction(
                instructions[pos], temperature
            )

        elif mutation_type == 'swap' and len(instructions) > 2:
            # Swap two instructions
            pos1 = random.randint(0, len(instructions) - 2)
            pos2 = random.randint(0, len(instructions) - 2)
            instructions[pos1], instructions[pos2] = instructions[pos2], instructions[pos1]

        elif mutation_type == 'pattern':
            # Insert a useful pattern
            instructions = self._insert_pattern(instructions, temperature)

        return Program(
            instructions=instructions,
            input_registers=input_registers,
            output_register=output_register
        )

    def _get_mutation_probabilities(self, temperature: float) -> List[float]:
        """Get mutation probabilities based on temperature."""
        # Normalize temperature to [0, 1] range
        t_norm = min(1.0, temperature / self.cooling_schedule.initial_temperature)

        # High temp: more add/remove, pattern insertion
        # Low temp: more modify, less structural changes
        return [
            0.2 + 0.2 * t_norm,    # add
            0.1 + 0.1 * t_norm,    # remove
            0.4 - 0.2 * t_norm,    # modify (more at low temp)
            0.15,                   # swap (constant)
            0.15 * t_norm,          # pattern (more at high temp)
        ]

    def _generate_structural_instruction(
        self,
        program_length: int,
        current_pos: int
    ) -> Instruction:
        """Generate instruction that adds structure (loops, conditionals)."""
        # Bias toward conditional branches at high temperature
        if random.random() < 0.5:
            # Conditional branch
            target = random.randint(0, program_length - 1)
            branch_type = random.choice([
                ARM64Opcode.BEQ, ARM64Opcode.BNE, ARM64Opcode.BLT,
                ARM64Opcode.BGE, ARM64Opcode.BGT, ARM64Opcode.BLE
            ])
            return Instruction(
                opcode=branch_type,
                branch_target=target
            )
        else:
            # CMP instruction to set up branch
            return Instruction(
                opcode=ARM64Opcode.CMP,
                rd=0,
                rn=random.randint(0, 5),
                rm=random.randint(0, 10),
                is_immediate=random.random() < 0.5
            )

    def _mutate_instruction(
        self,
        instr: Instruction,
        temperature: float
    ) -> Instruction:
        """Mutate a single instruction."""
        t_norm = min(1.0, temperature / self.cooling_schedule.initial_temperature)

        # At high temperature, might change opcode
        if t_norm > 0.5 and random.random() < 0.3:
            return self._random_instruction()

        # At low temperature, small tweaks
        field = random.choice(['rd', 'rn', 'rm'])

        if field == 'rd':
            new_rd = (instr.rd + random.randint(-2, 2)) % 11
            return Instruction(
                opcode=instr.opcode,
                rd=new_rd,
                rn=instr.rn,
                rm=instr.rm,
                is_immediate=instr.is_immediate,
                branch_target=instr.branch_target
            )
        elif field == 'rn':
            new_rn = (instr.rn + random.randint(-2, 2)) % 11
            return Instruction(
                opcode=instr.opcode,
                rd=instr.rd,
                rn=new_rn,
                rm=instr.rm,
                is_immediate=instr.is_immediate,
                branch_target=instr.branch_target
            )
        else:  # rm
            if instr.is_immediate:
                new_rm = max(0, min(100, instr.rm + random.randint(-5, 5)))
            else:
                new_rm = (instr.rm + random.randint(-2, 2)) % 11
            return Instruction(
                opcode=instr.opcode,
                rd=instr.rd,
                rn=instr.rn,
                rm=new_rm,
                is_immediate=instr.is_immediate,
                branch_target=instr.branch_target
            )

    def _insert_pattern(
        self,
        instructions: List[Instruction],
        temperature: float
    ) -> List[Instruction]:
        """Insert a useful code pattern."""
        patterns = [
            # Double: X0 = X0 + X0
            [make_add(0, 0, 0)],

            # Increment: X0 = X0 + 1
            [Instruction(ARM64Opcode.ADD, rd=0, rn=0, rm=1, is_immediate=True)],

            # Square setup: MOV X1, X0; MUL X0, X0, X1
            [make_mov_reg(1, 0), make_mul(0, 0, 1)],

            # Compare and branch pattern
            [make_cmp(0, 1), make_blt(0)],

            # Accumulator pattern: ADD X1, X1, X0
            [make_add(1, 1, 0)],
        ]

        pattern = random.choice(patterns)
        pos = random.randint(0, max(0, len(instructions) - 1))

        # Fix branch targets if needed
        for instr in pattern:
            if ARM64Opcode.is_branch(instr.opcode):
                instr.branch_target = max(0, min(len(instructions) - 1, instr.branch_target))

        new_instructions = instructions[:pos] + pattern + instructions[pos:]
        return new_instructions


# =============================================================================
# KVRM INTEGRATION
# =============================================================================

class KVRMAnnealingIntegration:
    """
    Integrates thermodynamic annealing with KVRM execution substrate.

    Uses KVRM for:
    1. Accurate program execution (100% accuracy)
    2. Energy computation with real execution results
    3. Verification of annealed programs
    """

    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.kvrm_available = False
        self.executor = None

        self._try_load_kvrm()

    def _try_load_kvrm(self):
        """Try to load KVRM integration."""
        try:
            from spnc.kvrm_integration import KVRMIntegration
            self.kvrm = KVRMIntegration(device=self.device, use_real_kvrm=False)
            self.kvrm_available = True
            print("KVRM integration loaded successfully")
        except ImportError as e:
            print(f"KVRM integration not available: {e}")
            print("Using simulated execution")
            self.kvrm_available = False

    def execute(self, program: Program, input_val: int) -> int:
        """Execute program using KVRM or simulation."""
        if self.kvrm_available:
            result = self.kvrm.execute_on_kvrm(program, {0: input_val})
            return result.output
        else:
            # Use built-in simulation
            return self._simulated_execute(program, input_val)

    def _simulated_execute(self, program: Program, input_val: int) -> int:
        """Fallback simulated execution."""
        registers = [0] * 32
        for reg in program.input_registers:
            registers[reg] = input_val

        pc = 0
        max_iterations = 1000
        iterations = 0
        flags = {'N': 0, 'Z': 0, 'C': 0, 'V': 0}

        while pc < len(program.instructions) and iterations < max_iterations:
            instr = program.instructions[pc]
            iterations += 1

            if instr.opcode == ARM64Opcode.RET:
                break
            elif instr.opcode == ARM64Opcode.ADD:
                b = instr.rm if instr.is_immediate else registers[min(instr.rm, 31)]
                registers[instr.rd] = (registers[instr.rn] + b) & ((1 << 64) - 1)
                pc += 1
            elif instr.opcode == ARM64Opcode.SUB:
                b = instr.rm if instr.is_immediate else registers[min(instr.rm, 31)]
                registers[instr.rd] = (registers[instr.rn] - b) & ((1 << 64) - 1)
                pc += 1
            elif instr.opcode == ARM64Opcode.MUL:
                b = instr.rm if instr.is_immediate else registers[min(instr.rm, 31)]
                registers[instr.rd] = (registers[instr.rn] * b) & ((1 << 64) - 1)
                pc += 1
            elif instr.opcode == ARM64Opcode.MOV:
                if instr.is_immediate:
                    registers[instr.rd] = instr.rm
                else:
                    registers[instr.rd] = registers[min(instr.rm, 31)]
                pc += 1
            elif instr.opcode == ARM64Opcode.CMP:
                a = registers[instr.rn]
                b = instr.rm if instr.is_immediate else registers[min(instr.rm, 31)]
                diff = a - b
                flags['N'] = 1 if diff < 0 else 0
                flags['Z'] = 1 if diff == 0 else 0
                flags['C'] = 1 if a >= b else 0
                pc += 1
            elif instr.opcode == ARM64Opcode.B:
                pc = instr.branch_target
            elif instr.opcode == ARM64Opcode.BEQ:
                pc = instr.branch_target if flags['Z'] else pc + 1
            elif instr.opcode == ARM64Opcode.BNE:
                pc = instr.branch_target if not flags['Z'] else pc + 1
            elif instr.opcode == ARM64Opcode.BLT:
                pc = instr.branch_target if flags['N'] else pc + 1
            elif instr.opcode == ARM64Opcode.BGE:
                pc = instr.branch_target if not flags['N'] else pc + 1
            elif instr.opcode == ARM64Opcode.BGT:
                pc = instr.branch_target if not flags['Z'] and not flags['N'] else pc + 1
            elif instr.opcode == ARM64Opcode.BLE:
                pc = instr.branch_target if flags['Z'] or flags['N'] else pc + 1
            else:
                pc += 1

        return registers[program.output_register]

    def verify_program(
        self,
        program: Program,
        test_cases: List[Tuple[int, int]]
    ) -> Tuple[bool, List[Tuple[int, int, int]]]:
        """Verify program against test cases."""
        results = []
        all_passed = True

        for input_val, expected in test_cases:
            actual = self.execute(program, input_val)
            passed = actual == expected
            results.append((input_val, expected, actual))
            if not passed:
                all_passed = False

        return all_passed, results


# =============================================================================
# DEMO AND TESTING
# =============================================================================

def demo_thermodynamic_annealing():
    """Demonstrate thermodynamic program annealing."""
    print("=" * 70)
    print("MOONSHOT #2: THERMODYNAMIC PROGRAM ANNEALING DEMO")
    print("=" * 70)
    print()

    # Test case: Find program that computes f(x) = 2x (doubling)
    test_cases = [
        (0, 0),
        (1, 2),
        (5, 10),
        (10, 20),
        (50, 100),
    ]

    print("Task: Synthesize program for f(x) = 2x")
    print(f"Test cases: {test_cases}")
    print()

    # Create annealer
    thermodynamics = ProgramThermodynamics(
        correctness_weight=100.0,
        length_weight=1.0,
        complexity_weight=0.5,
        semantic_weight=2.0
    )

    cooling = CoolingSchedule(
        schedule_type=CoolingScheduleType.EXPONENTIAL,
        initial_temperature=100.0,
        final_temperature=0.01,
        cooling_rate=0.997,
        max_steps=5000
    )

    annealer = ThermodynamicAnnealer(
        thermodynamics=thermodynamics,
        cooling_schedule=cooling
    )

    # Create KVRM integration for execution
    kvrm = KVRMAnnealingIntegration()

    # Run annealing
    result = annealer.anneal(
        test_cases=test_cases,
        input_registers=[0],
        output_register=0,
        max_program_length=10,
        executor=kvrm.execute,
        verbose=True
    )

    print()
    print("=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    if result.best_program:
        passed, verification = kvrm.verify_program(result.best_program, test_cases)
        print(f"Verification: {'PASSED' if passed else 'FAILED'}")
        for inp, expected, actual in verification:
            status = "OK" if expected == actual else "FAIL"
            print(f"  f({inp}) = {actual} (expected {expected}) [{status}]")

    return result


def demo_square_synthesis():
    """Demonstrate synthesizing square function f(x) = x*x."""
    print()
    print("=" * 70)
    print("CHALLENGE: Synthesize f(x) = x*x (Square)")
    print("=" * 70)
    print()

    test_cases = [
        (0, 0),
        (1, 1),
        (2, 4),
        (3, 9),
        (5, 25),
        (10, 100),
    ]

    print(f"Test cases: {test_cases}")
    print()

    # More aggressive annealing for harder problem
    thermodynamics = ProgramThermodynamics(
        correctness_weight=200.0,  # Higher weight for correctness
        length_weight=0.5,
        complexity_weight=0.3,
        semantic_weight=5.0  # Reward semantic patterns like square
    )

    cooling = CoolingSchedule(
        schedule_type=CoolingScheduleType.LOGARITHMIC,  # Slower cooling
        initial_temperature=200.0,
        final_temperature=0.01,
        max_steps=10000
    )

    annealer = ThermodynamicAnnealer(
        thermodynamics=thermodynamics,
        cooling_schedule=cooling
    )

    kvrm = KVRMAnnealingIntegration()

    result = annealer.anneal(
        test_cases=test_cases,
        input_registers=[0],
        output_register=0,
        max_program_length=8,
        executor=kvrm.execute,
        verbose=True
    )

    if result.best_program:
        print()
        print("Phase transitions detected:")
        for t in result.phase_transitions:
            print(f"  Step {t.step}: {t.from_phase.name} -> {t.to_phase.name} (T={t.temperature:.2f})")

    return result


if __name__ == "__main__":
    # Run demos
    result1 = demo_thermodynamic_annealing()

    print("\n" + "=" * 70 + "\n")

    result2 = demo_square_synthesis()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Double (f(x)=2x): {'Success' if result1.success else 'Failed'}")
    print(f"  Best energy: {result1.best_energy:.4f}")
    print(f"  Phase transitions: {len(result1.phase_transitions)}")
    print()
    print(f"Square (f(x)=x*x): {'Success' if result2.success else 'Failed'}")
    print(f"  Best energy: {result2.best_energy:.4f}")
    print(f"  Phase transitions: {len(result2.phase_transitions)}")
