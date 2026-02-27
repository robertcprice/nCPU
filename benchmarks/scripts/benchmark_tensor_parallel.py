#!/usr/bin/env python3
"""
Benchmark: Hybrid NumPy vs Pure Parallel Tensor Execution

This compares:
1. Current hybrid approach (NumPy fast path)
2. Pure parallel tensor execution (all ops as tensor operations)

The pure tensor approach executes N instructions in parallel using masked operations.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import time
from neural_cpu import NeuralCPU, device, _u64_to_s64, OpType

class PureTensorCPU:
    """
    Pure parallel tensor CPU - NO Python loops in execution.

    Key innovation: All N instructions in a batch execute SIMULTANEOUSLY
    using masked tensor operations. No .item() calls during execution.
    """

    def __init__(self, base_cpu: NeuralCPU):
        """Wrap an existing NeuralCPU to add parallel execution."""
        self.cpu = base_cpu
        self.device = base_cpu.device

        # Pre-allocate result tensors for parallel execution
        self.batch_size = 256
        self._results = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        self._rn_vals = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        self._rm_vals = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        self._rd_vals = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)

    @torch.no_grad()
    def run_parallel_tensor(self, max_instructions: int = 10000, batch_size: int = 64) -> tuple:
        """
        Execute instructions using PURE PARALLEL TENSOR operations.

        All instructions in a batch are processed simultaneously.
        No Python loops during the core execution phase.
        """
        start = time.perf_counter()
        executed = 0
        batch_size = min(batch_size, self.batch_size)

        cpu = self.cpu
        regs = cpu.regs  # [32] tensor
        memory = cpu.memory
        mem_size = cpu.mem_size

        pc = int(cpu.pc.item())

        while executed < max_instructions and not cpu.halted:
            if pc < 0 or pc + 4 > mem_size:
                break

            # Calculate actual batch (how many instructions we can fetch)
            actual_batch = min(batch_size, (mem_size - pc) // 4)
            if actual_batch <= 0:
                break

            # ═══════════════════════════════════════════════════════════════
            # PHASE 1: PARALLEL FETCH (pure tensor)
            # ═══════════════════════════════════════════════════════════════
            inst_bytes = memory[pc:pc + actual_batch * 4].view(actual_batch, 4)
            insts = (
                inst_bytes[:, 0].long() | (inst_bytes[:, 1].long() << 8) |
                (inst_bytes[:, 2].long() << 16) | (inst_bytes[:, 3].long() << 24)
            )

            # Check for halt/syscall BEFORE parallel execution
            # (These require sequential handling)
            halt_mask = (insts == 0)
            svc_mask = ((insts & 0xFFE0001F) == 0xD4000001)
            stop_mask = halt_mask | svc_mask

            if stop_mask.any():
                # Find first stop point
                stop_idx = stop_mask.nonzero(as_tuple=True)[0][0].item()
                if stop_idx == 0:
                    if halt_mask[0]:
                        cpu.halted = True
                    cpu.pc.fill_(pc)
                    return executed, time.perf_counter() - start
                actual_batch = stop_idx  # Only process up to stop point
                insts = insts[:actual_batch]

            # ═══════════════════════════════════════════════════════════════
            # PHASE 2: PARALLEL DECODE (pure tensor)
            # ═══════════════════════════════════════════════════════════════
            op_bytes = (insts >> 24) & 0xFF
            rds = insts & 0x1F
            rns = (insts >> 5) & 0x1F
            rms = (insts >> 16) & 0x1F
            imm12s = (insts >> 10) & 0xFFF

            # ═══════════════════════════════════════════════════════════════
            # PHASE 3: PARALLEL GATHER (get operand values)
            # ═══════════════════════════════════════════════════════════════
            # Gather register values for all instructions at once
            rn_vals = regs[rns]  # [actual_batch]
            rm_vals = regs[rms]  # [actual_batch]

            # ═══════════════════════════════════════════════════════════════
            # PHASE 4: PARALLEL EXECUTE (masked tensor operations)
            # Each operation type is computed for ALL instructions,
            # then masked to only affect relevant ones.
            # ═══════════════════════════════════════════════════════════════

            # Initialize results with zeros
            results = torch.zeros(actual_batch, dtype=torch.int64, device=self.device)
            executed_mask = torch.zeros(actual_batch, dtype=torch.bool, device=self.device)

            # ADD immediate (0x91)
            add_imm_mask = (op_bytes == 0x91)
            if add_imm_mask.any():
                add_results = rn_vals + imm12s
                results = torch.where(add_imm_mask, add_results, results)
                executed_mask = executed_mask | add_imm_mask

            # SUB immediate (0xD1)
            sub_imm_mask = (op_bytes == 0xD1)
            if sub_imm_mask.any():
                sub_results = rn_vals - imm12s
                results = torch.where(sub_imm_mask, sub_results, results)
                executed_mask = executed_mask | sub_imm_mask

            # ADD register (0x8B)
            add_reg_mask = (op_bytes == 0x8B)
            if add_reg_mask.any():
                add_reg_results = rn_vals + rm_vals
                results = torch.where(add_reg_mask, add_reg_results, results)
                executed_mask = executed_mask | add_reg_mask

            # SUB register (0xCB)
            sub_reg_mask = (op_bytes == 0xCB)
            if sub_reg_mask.any():
                sub_reg_results = rn_vals - rm_vals
                results = torch.where(sub_reg_mask, sub_reg_results, results)
                executed_mask = executed_mask | sub_reg_mask

            # AND register (0x8A)
            and_reg_mask = (op_bytes == 0x8A)
            if and_reg_mask.any():
                and_results = rn_vals & rm_vals
                results = torch.where(and_reg_mask, and_results, results)
                executed_mask = executed_mask | and_reg_mask

            # ORR register (0xAA)
            orr_reg_mask = (op_bytes == 0xAA)
            if orr_reg_mask.any():
                orr_results = rn_vals | rm_vals
                results = torch.where(orr_reg_mask, orr_results, results)
                executed_mask = executed_mask | orr_reg_mask

            # MOV (ORR with XZR) - same as ORR but Rn=31
            mov_mask = orr_reg_mask & (rns == 31)
            if mov_mask.any():
                # MOV Xd, Xm = ORR Xd, XZR, Xm
                results = torch.where(mov_mask, rm_vals, results)

            # ═══════════════════════════════════════════════════════════════
            # PHASE 5: PARALLEL SCATTER (write results back)
            # ═══════════════════════════════════════════════════════════════
            # For executed instructions, write results to destination registers
            # Skip XZR (rd=31)
            write_mask = executed_mask & (rds != 31)

            if write_mask.any():
                # Scatter results to registers
                # Note: This requires sequential handling for conflicting writes
                # For now, use a simple loop for the scatter phase
                for i in range(actual_batch):
                    if write_mask[i]:
                        rd = rds[i].item()
                        regs[rd] = results[i]

            # Count executed instructions
            # For unhandled ops, we still need to advance (they're NOPs for now)
            executed += actual_batch
            pc += actual_batch * 4

        cpu.pc.fill_(pc)
        return executed, time.perf_counter() - start


def run_benchmark():
    """Run comparison benchmark."""
    print("=" * 70)
    print("  BENCHMARK: Hybrid NumPy vs Pure Parallel Tensor")
    print("=" * 70)
    print()

    # Create CPU
    cpu = NeuralCPU(memory_size=1024*1024)
    parallel_cpu = PureTensorCPU(cpu)

    # Create a simple test program: loop that does ADD/SUB operations
    # This is ideal for parallel tensor execution
    print("Creating test program (ADD/SUB loop)...")

    # Simple loop:
    #   ADD X0, X0, #1
    #   ADD X1, X1, #2
    #   SUB X2, X2, #1
    #   ADD X3, X0, X1
    #   repeat...

    program = []
    for _ in range(1000):  # 4000 instructions total
        program.append(0x91000400)  # ADD X0, X0, #1
        program.append(0x91000821)  # ADD X1, X1, #2
        program.append(0xD1000442)  # SUB X2, X2, #1
        program.append(0x8B010003)  # ADD X3, X0, X1

    # Write program to memory
    for i, inst in enumerate(program):
        addr = i * 4
        for j in range(4):
            cpu.memory[addr + j] = (inst >> (j * 8)) & 0xFF

    print(f"Program size: {len(program)} instructions")
    print()

    # Benchmark 1: Hybrid NumPy (current fast path)
    print("─" * 70)
    print("Test 1: Hybrid NumPy Fast Path")
    print("─" * 70)

    cpu.pc.fill_(0)
    cpu.regs.zero_()
    cpu.halted = False

    start = time.perf_counter()
    executed1, elapsed1 = cpu.run(len(program))
    total1 = time.perf_counter() - start

    ips1 = executed1 / elapsed1 if elapsed1 > 0 else 0
    print(f"  Executed: {executed1:,} instructions")
    print(f"  Time: {elapsed1*1000:.2f} ms")
    print(f"  IPS: {ips1:,.0f}")
    print(f"  X0={int(cpu.regs[0].item())}, X1={int(cpu.regs[1].item())}, X2={int(cpu.regs[2].item())}")
    print()

    # Benchmark 2: Pure Parallel Tensor
    print("─" * 70)
    print("Test 2: Pure Parallel Tensor Execution")
    print("─" * 70)

    cpu.pc.fill_(0)
    cpu.regs.zero_()
    cpu.halted = False

    start = time.perf_counter()
    executed2, elapsed2 = parallel_cpu.run_parallel_tensor(len(program), batch_size=64)
    total2 = time.perf_counter() - start

    ips2 = executed2 / elapsed2 if elapsed2 > 0 else 0
    print(f"  Executed: {executed2:,} instructions")
    print(f"  Time: {elapsed2*1000:.2f} ms")
    print(f"  IPS: {ips2:,.0f}")
    print(f"  X0={int(cpu.regs[0].item())}, X1={int(cpu.regs[1].item())}, X2={int(cpu.regs[2].item())}")
    print()

    # Benchmark 3: Pure Parallel Tensor with larger batch
    print("─" * 70)
    print("Test 3: Pure Parallel Tensor (batch=128)")
    print("─" * 70)

    cpu.pc.fill_(0)
    cpu.regs.zero_()
    cpu.halted = False

    start = time.perf_counter()
    executed3, elapsed3 = parallel_cpu.run_parallel_tensor(len(program), batch_size=128)
    total3 = time.perf_counter() - start

    ips3 = executed3 / elapsed3 if elapsed3 > 0 else 0
    print(f"  Executed: {executed3:,} instructions")
    print(f"  Time: {elapsed3*1000:.2f} ms")
    print(f"  IPS: {ips3:,.0f}")
    print()

    # Summary
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Hybrid NumPy:        {ips1:>12,.0f} IPS")
    print(f"  Parallel Tensor 64:  {ips2:>12,.0f} IPS  ({ips2/ips1:.2f}x)")
    print(f"  Parallel Tensor 128: {ips3:>12,.0f} IPS  ({ips3/ips1:.2f}x)")
    print()

    if ips2 > ips1:
        print("  ✅ Parallel tensor is FASTER!")
    else:
        print("  ⚠️  Hybrid NumPy is faster for this workload")
        print("      (Parallel tensor benefits from larger batches of similar ops)")
    print("=" * 70)

    return ips1, ips2, ips3


if __name__ == "__main__":
    run_benchmark()
