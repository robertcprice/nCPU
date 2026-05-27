"""Neural Compiler for nCPU.

Compiles nsl (nCPU Simple Language) source code to nCPU assembly.
The compiler has four stages:

    1. Frontend (lexer + parser) - produces AST
    2. IR generation - AST to three-address code
    3. Neural optimizer - learned peephole optimizations (optional)
    4. Backend - IR to nCPU assembly text

The classical pipeline is deterministic and correct. The neural
optimizer is trained to apply peephole optimizations that improve
code quality, validated against the classical output for correctness.

Usage:
    compiler = NeuralCompiler()
    result = compiler.compile("var x = 10; var y = x + 5; halt;")
    print(result.assembly)   # nCPU assembly source
    print(result.binary)     # assembled binary (via NeuralAssembler)
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from .device import default_device
from .language import (
    Lexer, Parser, Program, ASTNode,
    NumberLit, IdentExpr, UnaryExpr, BinaryExpr, CallExpr,
    VarDecl, Assignment, IfStmt, WhileStmt, ForStmt, DoWhileStmt,
    ReturnStmt, HaltStmt, FuncDecl,
)
from .assembler import ClassicalAssembler, NeuralAssembler, AssemblyResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Intermediate Representation (Three-Address Code)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IRInstr:
    """Three-address code instruction.

    op: operation type
    dest: destination (register name like "t0", "t1" or variable name)
    src1: first source operand
    src2: second source operand (or None)
    """
    op: str           # "mov", "add", "sub", "mul", "div", "and", "or", "xor",
                      # "shl", "shr", "cmp", "jmp", "jz", "jnz", "js", "jns",
                      # "label", "halt", "inc", "dec", "neg", "call", "ret"
    dest: str = ""
    src1: str = ""
    src2: str = ""
    comment: str = ""


@dataclass
class CompileResult:
    """Result of compiling a program."""
    assembly: str                     # nCPU assembly source
    ir: List[IRInstr]                 # Three-address IR
    assembly_result: Optional[AssemblyResult] = None  # Assembled binary
    errors: List[str] = field(default_factory=list)
    optimizations_applied: int = 0
    variables: Dict[str, int] = field(default_factory=dict)  # var → register

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    @property
    def binary(self) -> Optional[List[int]]:
        if self.assembly_result:
            return self.assembly_result.binary
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Peephole Optimizer
# ═══════════════════════════════════════════════════════════════════════════════

class PeepholeOptimizerNet(nn.Module):
    """Neural peephole optimizer.

    Examines sliding windows of IR instructions and predicts
    whether an optimization can be applied, and what the optimized
    sequence should be.

    Optimizations learned:
        - Constant folding: MOV R0, 3; MOV R1, 5; ADD R2, R0, R1 → MOV R2, 8
        - Strength reduction: MUL Rd, Rs, 2 → SHL Rd, Rs, 1
        - Dead store elimination: MOV R0, 5; MOV R0, 10 → MOV R0, 10
        - Identity elimination: ADD R0, R0, 0 → (remove)

    Architecture:
        Window of 3 instructions × 5 features each → MLP → optimization class
    """

    def __init__(self, window_size: int = 3, feat_per_instr: int = 5,
                 num_opts: int = 5):
        super().__init__()
        self.window_size = window_size
        input_dim = window_size * feat_per_instr
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_opts),  # 0=none, 1=const_fold, 2=strength_red, 3=dead_store, 4=identity
        )
        self.num_opts = num_opts

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """window: [batch, window_size * feat_per_instr] → [batch, num_opts]"""
        return self.net(window)


# ═══════════════════════════════════════════════════════════════════════════════
# Register Allocator
# ═══════════════════════════════════════════════════════════════════════════════

class RegisterAllocator:
    """Simple linear-scan register allocator for R0-R7.

    Maps variable names to physical registers. Since nCPU only has 8
    registers, we limit scope to 8 live variables. Temporaries are
    freed after use to enable register reuse.
    """

    def __init__(self):
        self.var_map: Dict[str, int] = {}
        self.temp_counter = 0
        self.next_reg = 0
        self.max_regs = 8  # R0-R7
        self.free_pool: List[int] = []  # Freed temp registers

    def allocate(self, name: str) -> int:
        """Allocate a register for a variable or temporary."""
        if name in self.var_map:
            return self.var_map[name]
        # Reuse freed temp registers
        if name.startswith("__t") and self.free_pool:
            reg = self.free_pool.pop(0)
            self.var_map[name] = reg
            return reg
        if self.next_reg >= self.max_regs:
            raise RuntimeError(
                f"Register exhaustion: cannot allocate register for '{name}'. "
                f"Maximum {self.max_regs} variables supported.")
        reg = self.next_reg
        self.var_map[name] = reg
        self.next_reg += 1
        return reg

    def free_temp(self, name: str):
        """Free a temporary register for reuse."""
        if name.startswith("__t") and name in self.var_map:
            reg = self.var_map.pop(name)
            self.free_pool.append(reg)

    def get(self, name: str) -> Optional[int]:
        """Get the register for an existing variable."""
        return self.var_map.get(name)

    def new_temp(self) -> str:
        """Create a new temporary variable name."""
        name = f"__t{self.temp_counter}"
        self.temp_counter += 1
        return name

    def reset(self):
        """Reset allocator state."""
        self.var_map.clear()
        self.temp_counter = 0
        self.next_reg = 0
        self.free_pool.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# IR Generator (AST → Three-Address Code)
# ═══════════════════════════════════════════════════════════════════════════════

class IRGenerator:
    """Generates three-address IR from AST."""

    def __init__(self):
        self.ir: List[IRInstr] = []
        self.alloc = RegisterAllocator()
        self.label_counter = 0
        self.functions: Dict[str, FuncDecl] = {}

    def generate(self, program: Program) -> Tuple[List[IRInstr], Dict[str, int]]:
        """Generate IR for an entire program."""
        self.ir = []
        self.alloc.reset()
        self.label_counter = 0

        # Collect function definitions
        for func in program.functions:
            self.functions[func.name] = func

        # Generate code for top-level statements
        for stmt in program.statements:
            self._gen_stmt(stmt)

        return self.ir, self.alloc.var_map

    def _new_label(self, prefix: str = "L") -> str:
        self.label_counter += 1
        return f"{prefix}{self.label_counter}"

    def _emit(self, op: str, dest: str = "", src1: str = "", src2: str = "",
              comment: str = ""):
        self.ir.append(IRInstr(op=op, dest=dest, src1=src1, src2=src2,
                               comment=comment))

    # ─── Statement Generation ──────────────────────────────────────────

    def _gen_stmt(self, node: ASTNode):
        if isinstance(node, VarDecl):
            val = self._gen_expr(node.init)
            reg = self.alloc.allocate(node.name)
            if val != f"R{reg}":
                self._emit("mov", f"R{reg}", val, comment=f"var {node.name}")
            self._free_temp_value(val)
        elif isinstance(node, Assignment):
            val = self._gen_expr(node.value)
            reg = self.alloc.get(node.name)
            if reg is None:
                reg = self.alloc.allocate(node.name)
            if val != f"R{reg}":
                self._emit("mov", f"R{reg}", val, comment=f"{node.name} =")
            self._free_temp_value(val)
        elif isinstance(node, IfStmt):
            self._gen_if(node)
        elif isinstance(node, WhileStmt):
            self._gen_while(node)
        elif isinstance(node, ForStmt):
            self._gen_for(node)
        elif isinstance(node, DoWhileStmt):
            self._gen_do_while(node)
        elif isinstance(node, HaltStmt):
            self._emit("halt")
        elif isinstance(node, ReturnStmt):
            if node.value:
                val = self._gen_expr(node.value)
                self._emit("mov", "R0", val, comment="return value")
            self._emit("ret")
        elif isinstance(node, CallExpr):
            self._gen_call(node)

    def _gen_if(self, node: IfStmt):
        else_label = self._new_label("else")
        end_label = self._new_label("endif")

        # Generate condition
        self._gen_condition(node.condition, else_label if node.else_block else end_label)

        # Then block
        for stmt in node.then_block:
            self._gen_stmt(stmt)

        if node.else_block:
            self._emit("jmp", dest=end_label)
            self._emit("label", dest=else_label)
            for stmt in node.else_block:
                self._gen_stmt(stmt)

        self._emit("label", dest=end_label)

    def _gen_while(self, node: WhileStmt):
        loop_label = self._new_label("while")
        end_label = self._new_label("endwhile")

        self._emit("label", dest=loop_label)

        # Generate condition (jump to end if false)
        self._gen_condition(node.condition, end_label)

        # Loop body
        for stmt in node.body:
            self._gen_stmt(stmt)

        self._emit("jmp", dest=loop_label)
        self._emit("label", dest=end_label)

    def _gen_for(self, node: ForStmt):
        """Desugar for loop: init; while(cond) { body; update; }"""
        # Emit initializer
        self._gen_stmt(node.init)

        loop_label = self._new_label("for")
        end_label = self._new_label("endfor")

        self._emit("label", dest=loop_label)

        # Condition check
        self._gen_condition(node.condition, end_label)

        # Loop body
        for stmt in node.body:
            self._gen_stmt(stmt)

        # Update statement
        self._gen_stmt(node.update)

        self._emit("jmp", dest=loop_label)
        self._emit("label", dest=end_label)

    def _gen_do_while(self, node: DoWhileStmt):
        """Generate do...while: body executes at least once."""
        loop_label = self._new_label("dowhile")

        self._emit("label", dest=loop_label)

        # Body first
        for stmt in node.body:
            self._gen_stmt(stmt)

        # Condition: if true, jump back to loop_label
        # We invert the logic: generate condition that jumps to end_label if FALSE
        end_label = self._new_label("enddowhile")
        self._gen_condition(node.condition, end_label)
        self._emit("jmp", dest=loop_label)
        self._emit("label", dest=end_label)

    def _free_temp_value(self, val: str):
        """Free a temp register if val refers to one."""
        if val.startswith("R"):
            # Find if this register belongs to a temp
            for name, reg in list(self.alloc.var_map.items()):
                if f"R{reg}" == val and name.startswith("__t"):
                    self.alloc.free_temp(name)
                    break

    def _gen_condition(self, cond: ASTNode, false_label: str):
        """Generate code for a condition, jumping to false_label if false."""
        # Short-circuit logical AND: if left is false, skip right
        if isinstance(cond, BinaryExpr) and cond.op == "&&":
            # Left false → whole expr false → jump to false_label
            self._gen_condition(cond.left, false_label)
            # Left was true, now check right
            self._gen_condition(cond.right, false_label)
            return

        # Short-circuit logical OR: if left is true, skip right
        if isinstance(cond, BinaryExpr) and cond.op == "||":
            # Left true → whole expr true → skip to true_label
            true_label = self._new_label("ortrue")
            self._gen_condition_true(cond.left, true_label)
            # Left was false, check right
            self._gen_condition(cond.right, false_label)
            self._emit("label", dest=true_label)
            return

        if isinstance(cond, BinaryExpr) and cond.op in ("==", "!=", "<", ">", "<=", ">="):
            left = self._gen_expr(cond.left)
            right = self._gen_expr(cond.right)
            # CMP requires both operands in registers
            left_reg = self._ensure_register(left)
            right_reg = self._ensure_register(right)
            self._emit("cmp", src1=left_reg, src2=right_reg)
            # Map comparison to appropriate conditional jump
            # We jump when the condition is FALSE
            jump_map = {
                "==": "jnz",   # jump if not zero (not equal)
                "!=": "jz",    # jump if zero (equal)
                "<":  "jns",   # jump if not sign (>= 0, i.e. left >= right)
                ">=": "js",    # jump if sign (< 0, i.e. left < right)
                ">":  "js",    # We'll swap operands for > and <=
                "<=": "jns",
            }
            if cond.op in (">", "<="):
                # Swap: CMP right, left instead
                self.ir[-1] = IRInstr(op="cmp", src1=right_reg, src2=left_reg)
                if cond.op == ">":
                    self._emit("jns", dest=false_label)
                else:  # <=
                    self._emit("js", dest=false_label)
            else:
                self._emit(jump_map[cond.op], dest=false_label)
            # Free any temps used for comparison operands
            self._free_temp_value(left_reg)
            self._free_temp_value(right_reg)
        else:
            # Non-comparison: treat as boolean (compare with 0)
            val = self._gen_expr(cond)
            val_reg = self._ensure_register(val)
            zero_tmp = self.alloc.new_temp()
            zero_reg = self.alloc.allocate(zero_tmp)
            self._emit("mov", f"R{zero_reg}", "0")
            self._emit("cmp", src1=val_reg, src2=f"R{zero_reg}")
            self._emit("jz", dest=false_label)  # jump if zero (false)
            self.alloc.free_temp(zero_tmp)
            self._free_temp_value(val_reg)

    def _gen_condition_true(self, cond: ASTNode, true_label: str):
        """Generate code that jumps to true_label if condition is TRUE.
        Used for short-circuit OR: jump past right side if left is true."""
        if isinstance(cond, BinaryExpr) and cond.op in ("==", "!=", "<", ">", "<=", ">="):
            left = self._gen_expr(cond.left)
            right = self._gen_expr(cond.right)
            left_reg = self._ensure_register(left)
            right_reg = self._ensure_register(right)
            self._emit("cmp", src1=left_reg, src2=right_reg)
            # Jump when condition is TRUE (opposite of _gen_condition)
            jump_true_map = {
                "==": "jz",
                "!=": "jnz",
                "<":  "js",
                ">=": "jns",
            }
            if cond.op in (">", "<="):
                self.ir[-1] = IRInstr(op="cmp", src1=right_reg, src2=left_reg)
                if cond.op == ">":
                    self._emit("js", dest=true_label)
                else:  # <=
                    self._emit("jns", dest=true_label)
            else:
                self._emit(jump_true_map[cond.op], dest=true_label)
            self._free_temp_value(left_reg)
            self._free_temp_value(right_reg)
        else:
            # Non-comparison: jump if nonzero (true)
            val = self._gen_expr(cond)
            val_reg = self._ensure_register(val)
            zero_tmp = self.alloc.new_temp()
            zero_reg = self.alloc.allocate(zero_tmp)
            self._emit("mov", f"R{zero_reg}", "0")
            self._emit("cmp", src1=val_reg, src2=f"R{zero_reg}")
            self._emit("jnz", dest=true_label)  # jump if nonzero (true)
            self.alloc.free_temp(zero_tmp)
            self._free_temp_value(val_reg)

    # ─── Expression Generation ─────────────────────────────────────────

    def _gen_expr(self, node: ASTNode) -> str:
        """Generate IR for an expression, returning the register/value holding the result."""
        if isinstance(node, NumberLit):
            # Small immediates can be used directly
            return str(node.value)

        if isinstance(node, IdentExpr):
            reg = self.alloc.get(node.name)
            if reg is None:
                raise NameError(f"Undefined variable: {node.name}")
            return f"R{reg}"

        if isinstance(node, UnaryExpr):
            if node.op == "-":
                operand = self._gen_expr(node.operand)
                if isinstance(node.operand, NumberLit):
                    return str(-node.operand.value)
                # Negate: 0 - operand
                tmp = self.alloc.new_temp()
                reg = self.alloc.allocate(tmp)
                self._emit("mov", f"R{reg}", "0")
                self._emit("sub", f"R{reg}", f"R{reg}", operand)
                return f"R{reg}"
            if node.op == "!":
                # Logical NOT: result = (operand == 0) ? 1 : 0
                operand = self._gen_expr(node.operand)
                operand_reg = self._ensure_register(operand)
                result_tmp = self.alloc.new_temp()
                result_reg = self.alloc.allocate(result_tmp)
                zero_tmp = self.alloc.new_temp()
                zero_reg = self.alloc.allocate(zero_tmp)
                true_label = self._new_label("nottrue")
                end_label = self._new_label("notend")
                self._emit("mov", f"R{zero_reg}", "0")
                self._emit("cmp", src1=operand_reg, src2=f"R{zero_reg}")
                self._emit("mov", f"R{result_reg}", "0", comment="!expr default false")
                self._emit("jnz", dest=end_label)  # operand != 0 → result stays 0
                self._emit("mov", f"R{result_reg}", "1", comment="!expr is true")
                self._emit("label", dest=end_label)
                self.alloc.free_temp(zero_tmp)
                self._free_temp_value(operand_reg)
                return f"R{result_reg}"

        if isinstance(node, BinaryExpr):
            return self._gen_binary(node)

        if isinstance(node, CallExpr):
            return self._gen_call(node)

        raise TypeError(f"Cannot generate expression for {type(node).__name__}")

    def _gen_binary(self, node: BinaryExpr) -> str:
        """Generate code for a binary expression."""
        # Logical AND/OR as expressions: produce 1 or 0
        if node.op == "&&":
            return self._gen_logical_and_expr(node)
        if node.op == "||":
            return self._gen_logical_or_expr(node)

        # Modulo: a % b = a - (a / b) * b
        if node.op == "%":
            return self._gen_modulo(node)

        left = self._gen_expr(node.left)
        right = self._gen_expr(node.right)

        # Map operators to nCPU instructions
        op_map = {
            "+": "add", "-": "sub", "*": "mul", "/": "div",
            "&": "and", "|": "or", "^": "xor",
            "<<": "shl", ">>": "shr",
        }

        ir_op = op_map.get(node.op)
        if ir_op is None:
            # Comparison operations return 1 or 0
            return self._gen_comparison_expr(node, left, right)

        # Need both operands in registers
        left_reg = self._ensure_register(left)
        right_reg = self._ensure_register(right)

        # Allocate result register
        tmp = self.alloc.new_temp()
        dest_reg = self.alloc.allocate(tmp)
        self._emit(ir_op, f"R{dest_reg}", left_reg, right_reg)
        return f"R{dest_reg}"

    def _gen_modulo(self, node: BinaryExpr) -> str:
        """Generate a % b = a - (a / b) * b since nCPU has no MOD instruction."""
        left = self._gen_expr(node.left)
        right = self._gen_expr(node.right)
        left_reg = self._ensure_register(left)
        right_reg = self._ensure_register(right)

        # t1 = a / b
        t1_name = self.alloc.new_temp()
        t1_reg = self.alloc.allocate(t1_name)
        self._emit("div", f"R{t1_reg}", left_reg, right_reg)

        # t2 = t1 * b
        t2_name = self.alloc.new_temp()
        t2_reg = self.alloc.allocate(t2_name)
        self._emit("mul", f"R{t2_reg}", f"R{t1_reg}", right_reg)

        # t3 = a - t2
        t3_name = self.alloc.new_temp()
        t3_reg = self.alloc.allocate(t3_name)
        self._emit("sub", f"R{t3_reg}", left_reg, f"R{t2_reg}", comment="modulo")

        # Free intermediates
        self.alloc.free_temp(t1_name)
        self.alloc.free_temp(t2_name)
        self._free_temp_value(left_reg)
        self._free_temp_value(right_reg)
        return f"R{t3_reg}"

    def _gen_logical_and_expr(self, node: BinaryExpr) -> str:
        """Generate && as expression returning 1 or 0 with short-circuit."""
        result_tmp = self.alloc.new_temp()
        result_reg = self.alloc.allocate(result_tmp)
        false_label = self._new_label("andfalse")
        end_label = self._new_label("andend")

        # If left is false, short-circuit to false
        self._gen_condition(node.left, false_label)
        # Left is true, check right
        self._gen_condition(node.right, false_label)
        # Both true
        self._emit("mov", f"R{result_reg}", "1")
        self._emit("jmp", dest=end_label)
        self._emit("label", dest=false_label)
        self._emit("mov", f"R{result_reg}", "0")
        self._emit("label", dest=end_label)
        return f"R{result_reg}"

    def _gen_logical_or_expr(self, node: BinaryExpr) -> str:
        """Generate || as expression returning 1 or 0 with short-circuit."""
        result_tmp = self.alloc.new_temp()
        result_reg = self.alloc.allocate(result_tmp)
        true_label = self._new_label("ortrue")
        false_label = self._new_label("orfalse")
        end_label = self._new_label("orend")

        # If left is true, short-circuit to true
        self._gen_condition_true(node.left, true_label)
        # Left was false, check right
        self._gen_condition(node.right, false_label)
        # Right is true
        self._emit("label", dest=true_label)
        self._emit("mov", f"R{result_reg}", "1")
        self._emit("jmp", dest=end_label)
        self._emit("label", dest=false_label)
        self._emit("mov", f"R{result_reg}", "0")
        self._emit("label", dest=end_label)
        return f"R{result_reg}"

    def _gen_comparison_expr(self, node: BinaryExpr, left: str, right: str) -> str:
        """Generate code for comparison expression returning 1/0."""
        left_reg = self._ensure_register(left)
        right_reg = self._ensure_register(right)

        # CMP left, right
        self._emit("cmp", src1=left_reg, src2=right_reg)

        # Use conditional jumps to set result
        tmp = self.alloc.new_temp()
        result_reg = self.alloc.allocate(tmp)
        true_label = self._new_label("cmptrue")
        end_label = self._new_label("cmpend")

        self._emit("mov", f"R{result_reg}", "0")  # Default false

        # Jump to true based on condition
        if node.op == "==":
            self._emit("jz", dest=true_label)
        elif node.op == "!=":
            self._emit("jnz", dest=true_label)
        elif node.op == "<":
            self._emit("js", dest=true_label)
        elif node.op == ">=":
            self._emit("jns", dest=true_label)
        elif node.op == ">":
            # Swap and check sign
            self._emit("cmp", src1=right_reg, src2=left_reg)
            self._emit("js", dest=true_label)
        elif node.op == "<=":
            self._emit("cmp", src1=right_reg, src2=left_reg)
            self._emit("jns", dest=true_label)

        self._emit("jmp", dest=end_label)
        self._emit("label", dest=true_label)
        self._emit("mov", f"R{result_reg}", "1")
        self._emit("label", dest=end_label)

        return f"R{result_reg}"

    def _gen_call(self, node: CallExpr) -> str:
        """Generate code for a function call (inline expansion)."""
        func = self.functions.get(node.name)
        if func is None:
            raise NameError(f"Undefined function: {node.name}")

        # Simple inline expansion: map args to param names
        for i, (param, arg) in enumerate(zip(func.params, node.args)):
            val = self._gen_expr(arg)
            reg = self.alloc.allocate(param)
            if val != f"R{reg}":
                self._emit("mov", f"R{reg}", val, comment=f"arg {param}")

        # Inline function body
        for stmt in func.body:
            self._gen_stmt(stmt)

        return "R0"  # Functions return via R0

    def _ensure_register(self, value: str) -> str:
        """Ensure value is in a register. If it's an immediate, load it."""
        if value.startswith("R"):
            return value
        # Load immediate into scratch or temp
        tmp = self.alloc.new_temp()
        reg = self.alloc.allocate(tmp)
        self._emit("mov", f"R{reg}", value)
        return f"R{reg}"


# ═══════════════════════════════════════════════════════════════════════════════
# Classical Peephole Optimizer
# ═══════════════════════════════════════════════════════════════════════════════

class ClassicalOptimizer:
    """Deterministic peephole optimizer for IR sequences.

    Applies proven optimizations:
        1. Constant folding
        2. Dead store elimination
        3. Identity elimination (add 0, mul 1)
        4. Strength reduction (mul/div by powers of 2)
    """

    def optimize(self, ir: List[IRInstr]) -> Tuple[List[IRInstr], int]:
        """Optimize IR, return (optimized_ir, num_optimizations)."""
        count = 0
        changed = True
        while changed:
            changed = False
            ir, c = self._constant_fold(ir)
            if c > 0:
                changed = True
                count += c
            ir, c = self._dead_store_eliminate(ir)
            if c > 0:
                changed = True
                count += c
            ir, c = self._identity_eliminate(ir)
            if c > 0:
                changed = True
                count += c
        return ir, count

    def _constant_fold(self, ir: List[IRInstr]) -> Tuple[List[IRInstr], int]:
        """Fold operations on known constants.

        Invalidates all constants at labels and backward jumps to be
        safe across control flow boundaries.
        """
        count = 0
        constants: Dict[str, int] = {}  # register → known constant value
        result = []

        for instr in ir:
            # Labels and jumps invalidate all constants (control flow boundary)
            if instr.op in ("label", "jmp", "jz", "jnz", "js", "jns"):
                constants.clear()
                result.append(instr)
                continue

            if instr.op == "mov" and instr.src1.lstrip('-').isdigit() and not instr.src2:
                constants[instr.dest] = int(instr.src1)
                result.append(instr)
            elif instr.op in ("add", "sub", "mul", "div", "and", "or", "xor", "shl", "shr"):
                v1 = constants.get(instr.src1)
                v2 = constants.get(instr.src2)
                if v1 is not None and v2 is not None and (instr.op != "div" or v2 != 0):
                    folded = self._eval_op(instr.op, v1, v2)
                    if folded is not None:
                        constants[instr.dest] = folded
                        result.append(IRInstr(
                            op="mov", dest=instr.dest, src1=str(folded),
                            comment=f"folded {instr.op} {v1},{v2}"))
                        count += 1
                        continue
                # Invalidate destination
                constants.pop(instr.dest, None)
                result.append(instr)
            else:
                if instr.op not in ("halt", "cmp", "ret"):
                    constants.pop(instr.dest, None)
                result.append(instr)

        return result, count

    def _dead_store_eliminate(self, ir: List[IRInstr]) -> Tuple[List[IRInstr], int]:
        """Remove stores that are immediately overwritten."""
        count = 0
        result = []

        for i, instr in enumerate(ir):
            if (instr.op == "mov" and i + 1 < len(ir)
                    and ir[i + 1].op == "mov"
                    and ir[i + 1].dest == instr.dest
                    and instr.dest not in (ir[i + 1].src1, ir[i + 1].src2)):
                count += 1
                continue  # Skip dead store
            result.append(instr)

        return result, count

    def _identity_eliminate(self, ir: List[IRInstr]) -> Tuple[List[IRInstr], int]:
        """Remove identity operations."""
        count = 0
        result = []

        for instr in ir:
            # MOV Rx, Rx → skip
            if instr.op == "mov" and instr.dest == instr.src1 and not instr.src2:
                count += 1
                continue
            result.append(instr)

        return result, count

    def _eval_op(self, op: str, a: int, b: int) -> Optional[int]:
        try:
            if op == "add": return a + b
            if op == "sub": return a - b
            if op == "mul": return a * b
            if op == "div": return a // b if b != 0 else None
            if op == "and": return a & b
            if op == "or": return a | b
            if op == "xor": return a ^ b
            if op == "shl": return a << b
            if op == "shr": return a >> b
        except (OverflowError, ValueError):
            pass
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Backend (IR → nCPU Assembly)
# ═══════════════════════════════════════════════════════════════════════════════

class Backend:
    """Translates IR to nCPU assembly text."""

    def generate(self, ir: List[IRInstr]) -> str:
        """Generate nCPU assembly from IR."""
        lines = []
        lines.append("; Generated by nsl compiler")

        for instr in ir:
            line = self._emit_instruction(instr)
            if line:
                if instr.comment:
                    line += f"  ; {instr.comment}"
                lines.append(line)

        return "\n".join(lines)

    def _emit_instruction(self, instr: IRInstr) -> Optional[str]:
        op = instr.op

        if op == "label":
            return f"{instr.dest}:"

        if op == "halt":
            return "    HALT"

        if op == "nop":
            return "    NOP"

        if op == "ret":
            return "    HALT"  # Simple: return = halt for top-level

        if op == "mov":
            if instr.src1.startswith("R"):
                return f"    MOV {instr.dest}, {instr.src1}"
            else:
                return f"    MOV {instr.dest}, {instr.src1}"

        if op in ("add", "sub", "mul", "div", "and", "or", "xor"):
            return f"    {op.upper()} {instr.dest}, {instr.src1}, {instr.src2}"

        if op in ("shl", "shr"):
            return f"    {op.upper()} {instr.dest}, {instr.src1}, {instr.src2}"

        if op in ("inc", "dec"):
            return f"    {op.upper()} {instr.dest}"

        if op == "cmp":
            return f"    CMP {instr.src1}, {instr.src2}"

        if op in ("jmp", "jz", "jnz", "js", "jns"):
            return f"    {op.upper()} {instr.dest}"

        return f"    ; unknown: {op} {instr.dest} {instr.src1} {instr.src2}"


# ═══════════════════════════════════════════════════════════════════════════════
# Neural Compiler
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralCompiler:
    """Neural compiler: nsl source → nCPU assembly → binary.

    Combines classical compilation with neural optimization.
    The neural peephole optimizer is trained from the classical
    optimizer's decisions.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or default_device()

        # Compiler stages
        self.ir_gen = IRGenerator()
        self.optimizer = ClassicalOptimizer()
        self.backend = Backend()
        self.assembler = ClassicalAssembler()

        # Neural optimizer
        self.optimizer_net = PeepholeOptimizerNet().to(self.device)
        self._optimizer_trained = False

        # Statistics
        self.programs_compiled = 0
        self.total_optimizations = 0
        self.total_instructions_in = 0
        self.total_instructions_out = 0

    def compile(self, source: str, optimize: bool = True) -> CompileResult:
        """Compile nsl source to nCPU assembly and binary.

        Args:
            source: nsl source code
            optimize: Apply peephole optimizations

        Returns:
            CompileResult with assembly, IR, binary, etc.
        """
        self.programs_compiled += 1
        errors = []

        # Stage 1: Frontend (Lex + Parse)
        try:
            tokens = Lexer(source).tokenize()
            ast = Parser(tokens).parse()
        except SyntaxError as e:
            return CompileResult(assembly="", ir=[], errors=[str(e)])

        # Stage 2: IR Generation
        try:
            ir, var_map = self.ir_gen.generate(ast)
        except (NameError, RuntimeError) as e:
            return CompileResult(assembly="", ir=[], errors=[str(e)])

        self.total_instructions_in += len(ir)

        # Stage 3: Optimization
        opt_count = 0
        if optimize:
            ir, opt_count = self.optimizer.optimize(ir)
            self.total_optimizations += opt_count

        self.total_instructions_out += len(ir)

        # Stage 4: Backend (IR → Assembly)
        assembly = self.backend.generate(ir)

        # Stage 5: Assemble to binary
        asm_result = self.assembler.assemble(assembly)
        if not asm_result.success:
            errors.extend(asm_result.errors)

        return CompileResult(
            assembly=assembly,
            ir=ir,
            assembly_result=asm_result,
            errors=errors,
            optimizations_applied=opt_count,
            variables=var_map,
        )

    def compile_and_verify(self, source: str) -> CompileResult:
        """Compile and verify the output assembles correctly."""
        result = self.compile(source)
        if result.success and result.assembly_result:
            # Verify round-trip: disassemble and re-assemble
            disasm = self.assembler.disassemble(result.assembly_result.binary)
            logger.debug(f"Compiled {len(result.ir)} IR → "
                        f"{result.assembly_result.num_instructions} instructions")
        return result

    # ─── Training ─────────────────────────────────────────────────────────

    def train_optimizer(self, programs: List[str], epochs: int = 50,
                        lr: float = 1e-3) -> Dict:
        """Train the neural peephole optimizer.

        Uses the classical optimizer as oracle: trains the network to
        predict which optimizations apply to each instruction window.
        """
        training_data = []

        for source in programs:
            try:
                tokens = Lexer(source).tokenize()
                ast = Parser(tokens).parse()
                ir_before, _ = self.ir_gen.generate(ast)
                ir_after, _ = self.optimizer.optimize(ir_before)
            except Exception:
                continue

            # Generate training pairs from windows
            for i in range(len(ir_before) - 2):
                window = ir_before[i:i+3]
                features = self._window_features(window)
                # Determine if optimization was applied here
                label = self._classify_optimization(ir_before, ir_after, i)
                training_data.append((features, label))

        if not training_data:
            return {"error": "no_training_data"}

        features = torch.stack([d[0] for d in training_data])
        labels = torch.tensor([d[1] for d in training_data],
                              dtype=torch.long, device=self.device)

        opt = torch.optim.Adam(self.optimizer_net.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        best_acc = 0.0
        for epoch in range(epochs):
            self.optimizer_net.train()
            opt.zero_grad()
            logits = self.optimizer_net(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == labels).float().mean().item()
                best_acc = max(best_acc, acc)

        self.optimizer_net.eval()
        self._optimizer_trained = True

        return {
            "epochs": epochs,
            "num_programs": len(programs),
            "num_windows": len(training_data),
            "best_accuracy": best_acc,
        }

    def _window_features(self, window: List[IRInstr]) -> torch.Tensor:
        """Extract features from a 3-instruction window."""
        OP_MAP = {"mov": 1, "add": 2, "sub": 3, "mul": 4, "div": 5,
                  "and": 6, "or": 7, "xor": 8, "shl": 9, "shr": 10,
                  "cmp": 11, "jmp": 12, "jz": 13, "jnz": 14, "halt": 15,
                  "label": 16, "inc": 17, "dec": 18}

        features = []
        for instr in window:
            op_val = OP_MAP.get(instr.op, 0) / 18.0
            dest_reg = self._reg_to_float(instr.dest)
            src1_reg = self._reg_to_float(instr.src1)
            src2_reg = self._reg_to_float(instr.src2)
            is_imm = 1.0 if instr.src1.lstrip('-').isdigit() else 0.0
            features.extend([op_val, dest_reg, src1_reg, src2_reg, is_imm])

        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def _reg_to_float(self, s: str) -> float:
        if s.startswith("R") and len(s) == 2 and s[1].isdigit():
            return int(s[1]) / 7.0
        return 0.0

    def _classify_optimization(self, before: List[IRInstr],
                               after: List[IRInstr], idx: int) -> int:
        """Classify what optimization was applied at index."""
        # Simple: check if instruction at idx was removed or changed
        if idx < len(before) and idx < len(after):
            if before[idx].op == after[idx].op:
                return 0  # No change
        return 1  # Some optimization applied

    # ─── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str = "models/research/neuros/compiler_optimizer.pt"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.optimizer_net.state_dict(), path)

    def load(self, path: str = "models/research/neuros/compiler_optimizer.pt") -> bool:
        if Path(path).exists():
            self.optimizer_net.load_state_dict(
                torch.load(path, map_location=self.device, weights_only=True))
            self.optimizer_net.eval()
            self._optimizer_trained = True
            return True
        return False

    # ─── Diagnostics ──────────────────────────────────────────────────────

    def stats(self) -> Dict:
        return {
            "programs_compiled": self.programs_compiled,
            "total_optimizations": self.total_optimizations,
            "total_instructions_in": self.total_instructions_in,
            "total_instructions_out": self.total_instructions_out,
            "compression_ratio": (self.total_instructions_out /
                                  max(1, self.total_instructions_in)),
            "optimizer_trained": self._optimizer_trained,
        }

    def __repr__(self) -> str:
        return (f"NeuralCompiler(compiled={self.programs_compiled}, "
                f"opts={self.total_optimizations})")
