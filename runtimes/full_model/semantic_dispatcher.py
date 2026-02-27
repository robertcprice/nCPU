#!/usr/bin/env python3
"""
Semantic Dispatcher: Pattern-Based GPU Kernel Routing

Phase 2 of the "Intelligent Dispatcher" architecture. This module:

1. Detects high-level semantic patterns in instruction sequences
2. Routes to specialized GPU kernels that complete operations in ONE tensor op
3. Provides massive speedups for common patterns (memcpy, memset, strlen, etc.)

The key insight: Instead of emulating 1000 instructions for memcpy(dst, src, 1000),
we detect the pattern and execute `memory[dst:dst+1000] = memory[src:src+1000]` -
one GPU operation instead of thousands.

Supported patterns:
- memcpy: Bulk memory copy
- memset: Bulk memory fill
- strlen: String length computation
- strcmp: String comparison
- memcmp: Memory comparison
- array_sum: Array reduction
- linked_list: Pointer chasing (prefetch optimization)
"""

import torch
from typing import Optional, Tuple, Dict, List, Callable
from dataclasses import dataclass, field
from enum import Enum, auto


class SemanticOp(Enum):
    """High-level semantic operations we can accelerate."""
    MEMCPY = auto()
    MEMSET = auto()
    MEMMOVE = auto()
    STRLEN = auto()
    STRCMP = auto()
    MEMCMP = auto()
    STRCPY = auto()
    ARRAY_SUM = auto()
    ARRAY_MAX = auto()
    ARRAY_MIN = auto()
    LINKED_LIST_TRAVERSE = auto()
    UNKNOWN = auto()


@dataclass
class SemanticContext:
    """Context for a detected semantic operation."""
    op: SemanticOp
    src_addr: int = 0
    dst_addr: int = 0
    size: int = 0
    value: int = 0  # For memset
    stride: int = 1
    element_size: int = 1
    confidence: float = 0.0
    extra: Dict = field(default_factory=dict)


@dataclass
class DispatchResult:
    """Result of semantic dispatch."""
    handled: bool
    instructions_skipped: int = 0
    result_value: int = 0
    new_pc: int = 0
    registers_modified: Dict[int, int] = field(default_factory=dict)
    flags_modified: Dict[str, bool] = field(default_factory=dict)


class SemanticDispatcher:
    """
    Routes detected patterns to specialized GPU kernels.

    This is the brain of the semantic acceleration system. It:
    1. Maintains instruction sequence context
    2. Detects semantic patterns
    3. Routes to optimized kernels
    4. Returns results as if instructions executed normally
    """

    def __init__(
        self,
        memory: torch.Tensor,
        device=None,
        confidence_threshold: float = 0.8,
        detection_interval: int = 100,
        max_pattern_size: int = 10000
    ):
        self.device = device or memory.device
        self.memory = memory
        self.memory_size = len(memory)

        # ════════════════════════════════════════════════════════════════════
        # CONFIGURABLE THRESHOLDS (Phase 3 addition)
        # ════════════════════════════════════════════════════════════════════
        self.confidence_threshold = confidence_threshold  # Minimum confidence to trigger dispatch
        self.detection_interval = detection_interval       # Check patterns every N instructions
        self.max_pattern_size = max_pattern_size           # Maximum operations to accelerate at once

        # Throttling state
        self._instructions_since_check = 0
        self._last_dispatch_pc = -1
        self._consecutive_misses = 0
        self._adaptive_interval = detection_interval

        # Instruction sequence buffer for pattern detection
        self.inst_buffer: List[Tuple[int, int]] = []  # (pc, instruction)
        self.max_buffer = 32

        # Register state snapshot for pattern matching
        self.reg_snapshot: Dict[int, int] = {}

        # Pattern matchers (instruction sequence -> semantic op)
        self.pattern_matchers: List[Callable] = [
            self._match_memset_loop,
            self._match_memcpy_loop,
            self._match_strlen_loop,
            self._match_strcmp_loop,
            self._match_array_sum_loop,
        ]

        # Specialized kernels
        self.kernels: Dict[SemanticOp, Callable] = {
            SemanticOp.MEMSET: self._kernel_memset,
            SemanticOp.MEMCPY: self._kernel_memcpy,
            SemanticOp.MEMMOVE: self._kernel_memmove,
            SemanticOp.STRLEN: self._kernel_strlen,
            SemanticOp.STRCMP: self._kernel_strcmp,
            SemanticOp.MEMCMP: self._kernel_memcmp,
            SemanticOp.ARRAY_SUM: self._kernel_array_sum,
            SemanticOp.ARRAY_MAX: self._kernel_array_max,
            SemanticOp.ARRAY_MIN: self._kernel_array_min,
        }

        # Statistics
        self.stats = {
            'patterns_detected': 0,
            'instructions_accelerated': 0,
            'bytes_processed': 0,
            'kernel_calls': {op.name: 0 for op in SemanticOp},
            'try_dispatch_calls': 0,
            'detection_hits': 0,
            'detection_misses': 0,
        }

        # Enable/disable flag
        self.enabled = True

    def record_instruction(self, pc: int, instruction: int, regs: Dict[int, int]):
        """Record an instruction for pattern detection."""
        if not self.enabled:
            return

        self.inst_buffer.append((pc, instruction))
        if len(self.inst_buffer) > self.max_buffer:
            self.inst_buffer.pop(0)

        # Update register snapshot
        self.reg_snapshot = regs.copy()

    def try_dispatch(self, pc: int, regs: Dict[int, int]) -> Optional[DispatchResult]:
        """
        Try to detect a pattern and dispatch to a specialized kernel.

        Uses configurable confidence threshold and tracks detection statistics.

        Returns:
            DispatchResult if pattern detected and handled, None otherwise
        """
        if not self.enabled or len(self.inst_buffer) < 4:
            return None

        self.stats['try_dispatch_calls'] += 1

        # Avoid re-checking the same PC location repeatedly
        if pc == self._last_dispatch_pc:
            return None

        self._last_dispatch_pc = pc

        # Try each pattern matcher
        for matcher in self.pattern_matchers:
            context = matcher(pc, regs)
            if context and context.confidence >= self.confidence_threshold:
                self.stats['detection_hits'] += 1
                self._consecutive_misses = 0
                return self._execute_kernel(context, regs)

        # No pattern found
        self.stats['detection_misses'] += 1
        self._consecutive_misses += 1

        # Adaptive throttling: increase interval after consecutive misses
        if self._consecutive_misses > 10:
            self._adaptive_interval = min(self._adaptive_interval * 2, 1000)

        return None

    def should_check_patterns(self, instructions_executed: int) -> bool:
        """
        Check if we should attempt pattern detection (throttling).

        Called by execution loop to determine when to call try_dispatch().
        """
        self._instructions_since_check += instructions_executed

        if self._instructions_since_check >= self._adaptive_interval:
            self._instructions_since_check = 0
            return True

        return False

    def reset_throttling(self):
        """Reset adaptive throttling (e.g., at program boundaries)."""
        self._instructions_since_check = 0
        self._consecutive_misses = 0
        self._adaptive_interval = self.detection_interval
        self._last_dispatch_pc = -1

    def _validate_memory_range(self, start: int, size: int, op_name: str = 'operation') -> Tuple[bool, int, str]:
        """
        Validate a memory range for kernel operations.

        Performs comprehensive bounds checking to prevent:
        - Null page access (0x0-0x1000)
        - Buffer overflow beyond memory size
        - Negative addresses
        - Oversized operations (beyond max_pattern_size)

        Args:
            start: Start address
            size: Size of operation
            op_name: Operation name for logging

        Returns:
            (is_valid, clamped_size, rejection_reason)
        """
        # Initialize safety tracking
        if not hasattr(self, '_bounds_violations'):
            self._bounds_violations = {
                'null_page': 0,
                'overflow': 0,
                'underflow': 0,
                'oversized': 0,
                'total_rejected': 0,
                'total_validated': 0,
            }

        self._bounds_violations['total_validated'] += 1

        # Check 1: Null page protection
        if start < 0x1000:
            self._bounds_violations['null_page'] += 1
            self._bounds_violations['total_rejected'] += 1
            return False, 0, f'{op_name}: null page access at 0x{start:x}'

        # Check 2: Underflow protection
        if start < 0:
            self._bounds_violations['underflow'] += 1
            self._bounds_violations['total_rejected'] += 1
            return False, 0, f'{op_name}: negative address 0x{start:x}'

        # Check 3: Clamp size to memory bounds (graceful degradation)
        if start + size > self.memory_size:
            original_size = size
            size = max(0, self.memory_size - start)
            if size == 0:
                self._bounds_violations['overflow'] += 1
                self._bounds_violations['total_rejected'] += 1
                return False, 0, f'{op_name}: overflow at 0x{start:x} + {original_size}'

        # Check 4: Limit operation size (prevent runaway operations)
        if size > self.max_pattern_size:
            self._bounds_violations['oversized'] += 1
            size = self.max_pattern_size  # Clamp, don't reject

        return True, size, ''

    def get_bounds_violation_stats(self) -> dict:
        """Get statistics about bounds violations."""
        if not hasattr(self, '_bounds_violations'):
            return {'null_page': 0, 'overflow': 0, 'underflow': 0, 'oversized': 0, 'total_rejected': 0, 'total_validated': 0}
        return self._bounds_violations.copy()

    def force_dispatch(self, op: SemanticOp, **kwargs) -> Optional[DispatchResult]:
        """
        Force dispatch to a specific kernel (called by CPU when pattern is known).

        This is used when the CPU detects a libc call like memcpy() and wants
        to accelerate it directly without pattern matching.
        """
        if not self.enabled:
            return None

        context = SemanticContext(op=op, **kwargs)
        return self._execute_kernel(context, self.reg_snapshot)

    # ════════════════════════════════════════════════════════════════════════
    # PATTERN MATCHERS
    # ════════════════════════════════════════════════════════════════════════

    def _match_memset_loop(self, pc: int, regs: Dict[int, int]) -> Optional[SemanticContext]:
        """
        Detect memset loop pattern:

        Loop:
            STR XZR, [X0], #8   ; or STRB WZR, [X0], #1
            SUB X1, X1, #1      ; or #8
            CBNZ X1, Loop
        """
        if len(self.inst_buffer) < 3:
            return None

        # Look for store-with-post-increment + counter decrement + branch pattern
        recent = self.inst_buffer[-6:]

        store_found = False
        store_size = 0
        base_reg = -1
        count_reg = -1

        for _, inst in recent:
            op_byte = (inst >> 24) & 0xFF

            # Check for STR/STRB post-index with XZR/WZR
            if op_byte in [0xF8, 0x38]:  # STR/STRB post-index
                rt = inst & 0x1F
                if rt == 31:  # XZR/WZR
                    store_found = True
                    store_size = 8 if op_byte == 0xF8 else 1
                    base_reg = (inst >> 5) & 0x1F

            # Check for SUB with immediate
            if (inst >> 24) == 0xD1:  # SUB immediate
                rd = inst & 0x1F
                rn = (inst >> 5) & 0x1F
                if rd == rn and rd != base_reg:
                    count_reg = rd

        if store_found and count_reg >= 0 and base_reg >= 0:
            dst_addr = regs.get(base_reg, 0)
            count = regs.get(count_reg, 0)

            if count > 0 and 0 <= dst_addr < self.memory_size:
                return SemanticContext(
                    op=SemanticOp.MEMSET,
                    dst_addr=dst_addr,
                    size=count * store_size,
                    value=0,
                    stride=store_size,
                    confidence=0.9,
                    extra={'count_reg': count_reg, 'base_reg': base_reg}
                )

        return None

    def _match_memcpy_loop(self, pc: int, regs: Dict[int, int]) -> Optional[SemanticContext]:
        """
        Detect memcpy loop pattern:

        Loop:
            LDR X2, [X1], #8
            STR X2, [X0], #8
            SUB X3, X3, #1
            CBNZ X3, Loop
        """
        if len(self.inst_buffer) < 4:
            return None

        recent = self.inst_buffer[-8:]

        load_found = False
        store_found = False
        load_base = -1
        store_base = -1
        data_reg = -1
        count_reg = -1
        element_size = 0

        for _, inst in recent:
            op_byte = (inst >> 24) & 0xFF

            # LDR post-index
            if op_byte == 0xF8:
                idx_mode = (inst >> 10) & 0x3
                is_load = (inst >> 22) & 1
                if idx_mode == 0x1 and is_load:
                    load_found = True
                    load_base = (inst >> 5) & 0x1F
                    data_reg = inst & 0x1F
                    element_size = 8

            # STR post-index
            if op_byte == 0xF8:
                idx_mode = (inst >> 10) & 0x3
                is_load = (inst >> 22) & 1
                if idx_mode == 0x1 and not is_load:
                    rt = inst & 0x1F
                    if rt == data_reg:
                        store_found = True
                        store_base = (inst >> 5) & 0x1F

            # SUB immediate for counter
            if (inst >> 24) == 0xD1:
                rd = inst & 0x1F
                rn = (inst >> 5) & 0x1F
                if rd == rn and rd not in [load_base, store_base, data_reg]:
                    count_reg = rd

        if load_found and store_found and count_reg >= 0:
            src_addr = regs.get(load_base, 0)
            dst_addr = regs.get(store_base, 0)
            count = regs.get(count_reg, 0)

            if count > 0:
                return SemanticContext(
                    op=SemanticOp.MEMCPY,
                    src_addr=src_addr,
                    dst_addr=dst_addr,
                    size=count * element_size,
                    stride=element_size,
                    confidence=0.85,
                    extra={
                        'count_reg': count_reg,
                        'src_reg': load_base,
                        'dst_reg': store_base
                    }
                )

        return None

    def _match_strlen_loop(self, pc: int, regs: Dict[int, int]) -> Optional[SemanticContext]:
        """
        Detect strlen loop pattern:

        Loop:
            LDRB W1, [X0], #1
            ADD X2, X2, #1      ; counter
            CBNZ W1, Loop
        """
        if len(self.inst_buffer) < 3:
            return None

        recent = self.inst_buffer[-6:]

        byte_load_found = False
        base_reg = -1
        data_reg = -1
        counter_reg = -1

        for _, inst in recent:
            op_byte = (inst >> 24) & 0xFF

            # LDRB post-index
            if op_byte == 0x38:
                opc = (inst >> 22) & 0x3
                idx_mode = (inst >> 10) & 0x3
                if opc == 1 and idx_mode == 1:  # LDRB post-index
                    byte_load_found = True
                    base_reg = (inst >> 5) & 0x1F
                    data_reg = inst & 0x1F

            # ADD immediate (counter increment)
            if (inst >> 24) == 0x91:
                rd = inst & 0x1F
                rn = (inst >> 5) & 0x1F
                imm = (inst >> 10) & 0xFFF
                if rd == rn and imm == 1 and rd not in [base_reg, data_reg]:
                    counter_reg = rd

        if byte_load_found and counter_reg >= 0:
            str_addr = regs.get(base_reg, 0)

            if 0 <= str_addr < self.memory_size:
                return SemanticContext(
                    op=SemanticOp.STRLEN,
                    src_addr=str_addr,
                    confidence=0.85,
                    extra={
                        'base_reg': base_reg,
                        'counter_reg': counter_reg,
                        'data_reg': data_reg
                    }
                )

        return None

    def _match_strcmp_loop(self, pc: int, regs: Dict[int, int]) -> Optional[SemanticContext]:
        """
        Detect strcmp loop pattern:

        Loop:
            LDRB W2, [X0], #1
            LDRB W3, [X1], #1
            CMP W2, W3
            B.NE done
            CBNZ W2, Loop
        """
        if len(self.inst_buffer) < 5:
            return None

        recent = self.inst_buffer[-10:]

        load1_found = False
        load2_found = False
        base1 = -1
        base2 = -1

        for _, inst in recent:
            op_byte = (inst >> 24) & 0xFF

            # LDRB post-index
            if op_byte == 0x38:
                opc = (inst >> 22) & 0x3
                idx_mode = (inst >> 10) & 0x3
                if opc == 1 and idx_mode == 1:
                    if not load1_found:
                        load1_found = True
                        base1 = (inst >> 5) & 0x1F
                    else:
                        load2_found = True
                        base2 = (inst >> 5) & 0x1F

        if load1_found and load2_found:
            str1_addr = regs.get(base1, 0)
            str2_addr = regs.get(base2, 0)

            if 0 <= str1_addr < self.memory_size and 0 <= str2_addr < self.memory_size:
                return SemanticContext(
                    op=SemanticOp.STRCMP,
                    src_addr=str1_addr,
                    dst_addr=str2_addr,
                    confidence=0.8,
                    extra={'base1': base1, 'base2': base2}
                )

        return None

    def _match_array_sum_loop(self, pc: int, regs: Dict[int, int]) -> Optional[SemanticContext]:
        """
        Detect array sum loop pattern:

        Loop:
            LDR X2, [X0], #8
            ADD X1, X1, X2      ; accumulator
            SUB X3, X3, #1
            CBNZ X3, Loop
        """
        if len(self.inst_buffer) < 4:
            return None

        recent = self.inst_buffer[-8:]

        load_found = False
        add_found = False
        base_reg = -1
        data_reg = -1
        accum_reg = -1
        count_reg = -1
        element_size = 0

        for _, inst in recent:
            op_byte = (inst >> 24) & 0xFF

            # LDR post-index
            if op_byte == 0xF8:
                idx_mode = (inst >> 10) & 0x3
                is_load = (inst >> 22) & 1
                if idx_mode == 0x1 and is_load:
                    load_found = True
                    base_reg = (inst >> 5) & 0x1F
                    data_reg = inst & 0x1F
                    element_size = 8

            # ADD register (accumulator)
            if (inst >> 24) == 0x8B:  # ADD Xd, Xn, Xm
                rd = inst & 0x1F
                rn = (inst >> 5) & 0x1F
                rm = (inst >> 16) & 0x1F
                if rm == data_reg and rd == rn:
                    add_found = True
                    accum_reg = rd

            # SUB immediate for counter
            if (inst >> 24) == 0xD1:
                rd = inst & 0x1F
                rn = (inst >> 5) & 0x1F
                if rd == rn and rd not in [base_reg, data_reg, accum_reg]:
                    count_reg = rd

        if load_found and add_found and count_reg >= 0:
            arr_addr = regs.get(base_reg, 0)
            count = regs.get(count_reg, 0)

            if count > 0 and 0 <= arr_addr < self.memory_size:
                return SemanticContext(
                    op=SemanticOp.ARRAY_SUM,
                    src_addr=arr_addr,
                    size=count * element_size,
                    element_size=element_size,
                    confidence=0.85,
                    extra={
                        'base_reg': base_reg,
                        'accum_reg': accum_reg,
                        'count_reg': count_reg
                    }
                )

        return None

    # ════════════════════════════════════════════════════════════════════════
    # SPECIALIZED GPU KERNELS
    # ════════════════════════════════════════════════════════════════════════

    def _execute_kernel(self, context: SemanticContext, regs: Dict[int, int]) -> Optional[DispatchResult]:
        """Execute the appropriate kernel for the detected pattern."""
        kernel = self.kernels.get(context.op)
        if kernel is None:
            return None

        result = kernel(context, regs)

        if result and result.handled:
            self.stats['patterns_detected'] += 1
            self.stats['instructions_accelerated'] += result.instructions_skipped
            self.stats['kernel_calls'][context.op.name] += 1
            if context.size > 0:
                self.stats['bytes_processed'] += context.size

        return result

    def _kernel_memset(self, ctx: SemanticContext, regs: Dict[int, int]) -> DispatchResult:
        """
        GPU-accelerated memset.

        Instead of: N iterations of STR XZR, [Xn], #8
        Execute: memory[dst:dst+size] = value (ONE tensor operation)
        """
        dst = ctx.dst_addr
        size = ctx.size
        value = ctx.value

        # Enhanced bounds validation (Phase 3 safety)
        is_valid, size, reason = self._validate_memory_range(dst, size, 'memset')
        if not is_valid:
            return DispatchResult(handled=False)

        # ONE tensor operation to fill memory
        self.memory[dst:dst + size] = value

        # Update registers as if loop completed
        count_reg = ctx.extra.get('count_reg', -1)
        base_reg = ctx.extra.get('base_reg', -1)

        reg_mods = {}
        if count_reg >= 0:
            reg_mods[count_reg] = 0  # Counter exhausted
        if base_reg >= 0:
            reg_mods[base_reg] = dst + size  # Base advanced

        # Estimate instructions skipped: ~3 per iteration
        iterations = size // max(ctx.stride, 1)

        return DispatchResult(
            handled=True,
            instructions_skipped=iterations * 3,
            registers_modified=reg_mods
        )

    def _kernel_memcpy(self, ctx: SemanticContext, regs: Dict[int, int]) -> DispatchResult:
        """
        GPU-accelerated memcpy.

        Instead of: N iterations of LDR + STR
        Execute: memory[dst:dst+size] = memory[src:src+size] (ONE tensor operation)
        """
        src = ctx.src_addr
        dst = ctx.dst_addr
        size = ctx.size

        # Enhanced bounds validation for source (Phase 3 safety)
        is_valid_src, size, reason = self._validate_memory_range(src, size, 'memcpy_src')
        if not is_valid_src:
            return DispatchResult(handled=False)

        # Enhanced bounds validation for destination
        is_valid_dst, size, reason = self._validate_memory_range(dst, size, 'memcpy_dst')
        if not is_valid_dst:
            return DispatchResult(handled=False)

        # Check for overlap (need memmove semantics)
        if src < dst < src + size:
            # Overlapping, copy backwards - use clone to avoid aliasing
            self.memory[dst:dst + size] = self.memory[src:src + size].clone()
        else:
            # Non-overlapping, direct copy
            self.memory[dst:dst + size] = self.memory[src:src + size]

        # Update registers
        reg_mods = {}
        count_reg = ctx.extra.get('count_reg', -1)
        src_reg = ctx.extra.get('src_reg', -1)
        dst_reg = ctx.extra.get('dst_reg', -1)

        if count_reg >= 0:
            reg_mods[count_reg] = 0
        if src_reg >= 0:
            reg_mods[src_reg] = src + size
        if dst_reg >= 0:
            reg_mods[dst_reg] = dst + size

        iterations = size // max(ctx.stride, 1)

        return DispatchResult(
            handled=True,
            instructions_skipped=iterations * 4,  # LDR + STR + SUB + CBNZ
            registers_modified=reg_mods
        )

    def _kernel_memmove(self, ctx: SemanticContext, regs: Dict[int, int]) -> DispatchResult:
        """GPU-accelerated memmove (handles overlapping regions)."""
        # Same as memcpy but always use clone for safety
        src = ctx.src_addr
        dst = ctx.dst_addr
        size = ctx.size

        if src < 0 or src + size > self.memory_size:
            return DispatchResult(handled=False)
        if dst < 0 or dst + size > self.memory_size:
            return DispatchResult(handled=False)

        # Always clone to handle any overlap
        self.memory[dst:dst + size] = self.memory[src:src + size].clone()

        return DispatchResult(
            handled=True,
            instructions_skipped=(size // 8) * 4
        )

    def _kernel_strlen(self, ctx: SemanticContext, regs: Dict[int, int]) -> DispatchResult:
        """
        GPU-accelerated strlen.

        Instead of: Loop checking each byte
        Execute: Find first zero in tensor (vectorized search)
        """
        addr = ctx.src_addr

        if addr < 0 or addr >= self.memory_size:
            return DispatchResult(handled=False)

        # Vectorized null terminator search
        max_len = min(self.memory_size - addr, 65536)  # Cap search
        chunk = self.memory[addr:addr + max_len]

        # Find first zero
        zero_mask = (chunk == 0)
        if zero_mask.any():
            length = int(zero_mask.int().argmax().item())
        else:
            length = max_len

        # Update registers
        reg_mods = {}
        counter_reg = ctx.extra.get('counter_reg', -1)
        base_reg = ctx.extra.get('base_reg', -1)

        if counter_reg >= 0:
            reg_mods[counter_reg] = length
        if base_reg >= 0:
            reg_mods[base_reg] = addr + length + 1

        return DispatchResult(
            handled=True,
            instructions_skipped=length * 3,  # LDRB + ADD + CBNZ per char
            result_value=length,
            registers_modified=reg_mods
        )

    def _kernel_strcmp(self, ctx: SemanticContext, regs: Dict[int, int]) -> DispatchResult:
        """
        GPU-accelerated strcmp.

        Vectorized string comparison.
        """
        addr1 = ctx.src_addr
        addr2 = ctx.dst_addr

        if addr1 < 0 or addr2 < 0:
            return DispatchResult(handled=False)

        max_len = min(
            self.memory_size - addr1,
            self.memory_size - addr2,
            65536
        )

        str1 = self.memory[addr1:addr1 + max_len]
        str2 = self.memory[addr2:addr2 + max_len]

        # Find first difference or null
        diff_mask = (str1 != str2) | (str1 == 0)

        if diff_mask.any():
            idx = int(diff_mask.int().argmax().item())
            val1 = int(str1[idx].item())
            val2 = int(str2[idx].item())
            result = val1 - val2
        else:
            result = 0
            idx = max_len

        return DispatchResult(
            handled=True,
            instructions_skipped=idx * 5,  # 2x LDRB + CMP + 2x branch
            result_value=result
        )

    def _kernel_memcmp(self, ctx: SemanticContext, regs: Dict[int, int]) -> DispatchResult:
        """GPU-accelerated memcmp."""
        addr1 = ctx.src_addr
        addr2 = ctx.dst_addr
        size = ctx.size

        if addr1 < 0 or addr1 + size > self.memory_size:
            return DispatchResult(handled=False)
        if addr2 < 0 or addr2 + size > self.memory_size:
            return DispatchResult(handled=False)

        mem1 = self.memory[addr1:addr1 + size]
        mem2 = self.memory[addr2:addr2 + size]

        diff_mask = (mem1 != mem2)

        if diff_mask.any():
            idx = int(diff_mask.int().argmax().item())
            result = int(mem1[idx].item()) - int(mem2[idx].item())
        else:
            result = 0

        return DispatchResult(
            handled=True,
            instructions_skipped=size * 3,
            result_value=result
        )

    def _kernel_array_sum(self, ctx: SemanticContext, regs: Dict[int, int]) -> DispatchResult:
        """
        GPU-accelerated array sum.

        Instead of: Loop loading and adding each element
        Execute: tensor.sum() (ONE operation)
        """
        addr = ctx.src_addr
        size = ctx.size
        elem_size = ctx.element_size

        if addr < 0 or addr + size > self.memory_size:
            return DispatchResult(handled=False)

        num_elements = size // elem_size

        # Load as appropriate dtype and sum
        if elem_size == 8:
            # Load as int64
            raw = self.memory[addr:addr + size]
            # Reshape and interpret as int64
            values = raw.view(torch.uint8).reshape(-1, 8)
            # Manual int64 reconstruction
            total = 0
            for i in range(num_elements):
                val = 0
                for j in range(8):
                    val |= int(values[i, j].item()) << (j * 8)
                # Handle signed
                if val >= 2**63:
                    val -= 2**64
                total += val
        else:
            # Simpler case for smaller elements
            raw = self.memory[addr:addr + size]
            total = int(raw.sum().item())

        reg_mods = {}
        accum_reg = ctx.extra.get('accum_reg', -1)
        count_reg = ctx.extra.get('count_reg', -1)
        base_reg = ctx.extra.get('base_reg', -1)

        if accum_reg >= 0:
            # Add to existing accumulator value
            existing = regs.get(accum_reg, 0)
            reg_mods[accum_reg] = existing + total
        if count_reg >= 0:
            reg_mods[count_reg] = 0
        if base_reg >= 0:
            reg_mods[base_reg] = addr + size

        return DispatchResult(
            handled=True,
            instructions_skipped=num_elements * 4,
            result_value=total,
            registers_modified=reg_mods
        )

    def _kernel_array_max(self, ctx: SemanticContext, regs: Dict[int, int]) -> DispatchResult:
        """GPU-accelerated array max."""
        addr = ctx.src_addr
        size = ctx.size

        if addr < 0 or addr + size > self.memory_size:
            return DispatchResult(handled=False)

        raw = self.memory[addr:addr + size]
        max_val = int(raw.max().item())

        return DispatchResult(
            handled=True,
            instructions_skipped=size * 3,
            result_value=max_val
        )

    def _kernel_array_min(self, ctx: SemanticContext, regs: Dict[int, int]) -> DispatchResult:
        """GPU-accelerated array min."""
        addr = ctx.src_addr
        size = ctx.size

        if addr < 0 or addr + size > self.memory_size:
            return DispatchResult(handled=False)

        raw = self.memory[addr:addr + size]
        min_val = int(raw.min().item())

        return DispatchResult(
            handled=True,
            instructions_skipped=size * 3,
            result_value=min_val
        )

    # ════════════════════════════════════════════════════════════════════════
    # STATISTICS
    # ════════════════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict:
        """Get dispatch statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'patterns_detected': 0,
            'instructions_accelerated': 0,
            'bytes_processed': 0,
            'kernel_calls': {op.name: 0 for op in SemanticOp},
        }

    def print_stats(self):
        """Print dispatch statistics."""
        print("\n" + "=" * 50)
        print("  SEMANTIC DISPATCHER STATISTICS")
        print("=" * 50)
        print(f"  Patterns detected: {self.stats['patterns_detected']:,}")
        print(f"  Instructions accelerated: {self.stats['instructions_accelerated']:,}")
        print(f"  Bytes processed: {self.stats['bytes_processed']:,}")
        print("\n  Kernel calls:")
        for name, count in self.stats['kernel_calls'].items():
            if count > 0:
                print(f"    {name}: {count:,}")
        print("=" * 50)


# ════════════════════════════════════════════════════════════════════════════════
# QUICK TEST
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  SEMANTIC DISPATCHER TEST")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create test memory
    memory = torch.zeros(1024 * 1024, dtype=torch.uint8, device=device)
    dispatcher = SemanticDispatcher(memory, device=device)

    # Test 1: Force dispatch memset
    print("\n[TEST 1] Force memset")
    result = dispatcher.force_dispatch(
        SemanticOp.MEMSET,
        dst_addr=0x1000,
        size=1000,
        value=0x42,
        stride=1
    )
    print(f"  Handled: {result.handled}")
    print(f"  Instructions skipped: {result.instructions_skipped}")
    print(f"  Memory[0x1000]: {memory[0x1000].item()} (expected: 0x42)")

    # Test 2: Force dispatch memcpy
    print("\n[TEST 2] Force memcpy")
    # Setup source data
    for i in range(100):
        memory[0x2000 + i] = i & 0xFF

    result = dispatcher.force_dispatch(
        SemanticOp.MEMCPY,
        src_addr=0x2000,
        dst_addr=0x3000,
        size=100,
        stride=1
    )
    print(f"  Handled: {result.handled}")
    print(f"  Instructions skipped: {result.instructions_skipped}")
    print(f"  Memory[0x3050]: {memory[0x3050].item()} (expected: 50)")

    # Test 3: Force dispatch strlen
    print("\n[TEST 3] Force strlen")
    # Setup string
    test_str = b"Hello, World!"
    for i, c in enumerate(test_str):
        memory[0x4000 + i] = c
    memory[0x4000 + len(test_str)] = 0  # Null terminator

    result = dispatcher.force_dispatch(
        SemanticOp.STRLEN,
        src_addr=0x4000
    )
    print(f"  Handled: {result.handled}")
    print(f"  Length: {result.result_value} (expected: {len(test_str)})")

    # Test 4: Force dispatch strcmp
    print("\n[TEST 4] Force strcmp")
    str1 = b"Hello"
    str2 = b"Hello"
    for i, c in enumerate(str1):
        memory[0x5000 + i] = c
    memory[0x5000 + len(str1)] = 0
    for i, c in enumerate(str2):
        memory[0x5100 + i] = c
    memory[0x5100 + len(str2)] = 0

    result = dispatcher.force_dispatch(
        SemanticOp.STRCMP,
        src_addr=0x5000,
        dst_addr=0x5100
    )
    print(f"  Handled: {result.handled}")
    print(f"  Result: {result.result_value} (expected: 0 for equal)")

    # Print stats
    dispatcher.print_stats()

    print("\n[DONE] Semantic Dispatcher tests complete!")
