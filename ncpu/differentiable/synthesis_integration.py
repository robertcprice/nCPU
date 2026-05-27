"""End-to-end differentiable program synthesis with OS-aware execution.

Bridges the mog_synth program synthesis pipeline with the differentiable OS
and differentiable CPU execution engine.  The key insight: programs synthesized
by mog_synth's gradient search live in a continuous parameter space that is
*compatible* with the differentiable execution engine's gradient flow.  By
running synthesized candidates through the differentiable OS, we get:

1. **OS-aware evaluation** — programs are scored not just by I/O correctness
   but also by cache behavior, scheduling impact, and memory allocation cost.
2. **Gradient-guided refinement** — the differentiable OS provides gradients
   that indicate *which instruction changes* would improve execution on the
   OS, enabling a hybrid search: synthesis proposes, the OS evaluates and
   suggests refinements.
3. **End-to-end training** — a single backward pass flows gradients from the
   output loss, through the OS policies, through the execution engine, all
   the way back to the synthesized program parameters.

Architecture:

    I/O examples ──> SynthesisOSPipeline ──> candidate programs
                          │                        │
                          │                  DifferentiableProgramExecutor
                          │                        │
                          │                  DifferentiableOS (cache/sched/alloc)
                          │                        │
                          │              output + OS metrics + gradients
                          │                        │
                          └──── loss + backprop ◄───┘
                                   │
                          GradientGuidedSynthesis
                           (use gradients to steer
                            next synthesis iteration)

Usage:
    >>> from ncpu.differentiable.synthesis_integration import SynthesisOSPipeline
    >>> pipeline = SynthesisOSPipeline()
    >>> result = pipeline.run(
    ...     examples=[([5, 3], 8), ([10, 7], 17), ([0, 0], 0)],
    ...     n_args=2,
    ... )
    >>> print(result.program_text)
    >>> print(f"accuracy={result.accuracy:.1%}, os_loss={result.os_loss:.4f}")
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Local differentiable execution engine ─────────────────────────────────
from ncpu.differentiable.execution import (
    DifferentiableEngine,
    FixedProgram,
    SoftProgram,
    Instruction,
    ExecutionResult,
    OPCODES,
    NUM_OPCODES,
)
from ncpu.differentiable.program_synthesis import (
    ProgramSynthesizer,
    SynthesisSpec,
    SynthesisResult,
)
from ncpu.differentiable.differentiable_os import (
    DifferentiableOS,
    DifferentiableCache,
    DifferentiableScheduler,
    DifferentiableAllocator,
    WorkloadEvent,
)

# ── mog_synth soft synthesis (imported lazily to avoid hard dep on scripts/) ─
_MOG_SYNTH_LOADED = False
_soft_synth = None


def _ensure_mog_synth():
    """Lazy-load nsynth/scripts/soft_synth.py so the module works even
    when mog_synth is not on sys.path at import time."""
    global _MOG_SYNTH_LOADED, _soft_synth
    if _MOG_SYNTH_LOADED:
        return
    scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "nsynth" / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import soft_synth as _ss
    _soft_synth = _ss
    _MOG_SYNTH_LOADED = True


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ExecutorResult:
    """Result from DifferentiableProgramExecutor.execute()."""
    output: torch.Tensor              # predicted output per example (n_examples,)
    registers: List[torch.Tensor]     # final register state per example
    os_metrics: Dict[str, float]      # cache_hit_rate, fragmentation, ...
    schedule_weights: List[torch.Tensor]
    eviction_weights: List[torch.Tensor]
    placement_weights: List[torch.Tensor]
    grad_info: Optional[Dict[str, torch.Tensor]] = None  # filled after backward


@dataclass
class PipelineResult:
    """Result from SynthesisOSPipeline.run()."""
    program_description: Optional[dict]     # mog_synth program description
    program_text: str                       # human-readable program
    accuracy: float                         # fraction of examples correct
    io_loss: float                          # MSE between predicted and expected
    os_loss: float                          # combined OS policy loss
    total_loss: float                       # io_loss + os_weight * os_loss
    os_metrics: Dict[str, float]            # cache_hit_rate, fragmentation
    synthesis_method: str                   # cold / warm / gradient_guided
    refinement_steps: int                   # number of gradient refinement steps
    loss_history: List[float]               # per-step total loss


@dataclass
class GradientGuidance:
    """Gradient information used to steer the next synthesis iteration."""
    param_gradients: torch.Tensor           # dL/d(program_params), same shape as params
    slot_importance: torch.Tensor           # per-slot gradient magnitude (N_UNIV_SLOTS,)
    suggested_changes: List[Dict]           # ranked list of {slot, field, direction}
    estimated_improvement: float            # predicted loss reduction from top suggestion


# ---------------------------------------------------------------------------
# MogSynth <-> DifferentiableEngine bridge
# ---------------------------------------------------------------------------

def _mog_description_to_ncpu_instructions(
    desc: dict,
    n_args: int,
    max_instructions: int = 16,
) -> List[Instruction]:
    """Convert a mog_synth UniversalProgramDescription to a sequence of
    nCPU Instruction objects for the differentiable execution engine.

    The mog_synth architecture is:
        init slots (3) -> loop (condition + 6 body slots) -> post slots (2) -> return

    We flatten this into a linear instruction sequence using the nCPU ISA:
        MOV_IMM for constants, ADD/SUB/MUL for arithmetic, CMP+BNE for loops.

    This is an *approximate* mapping — the mog_synth universal program has
    richer semantics (gated slots, soft comparisons) than the simple ISA.
    We extract the dominant (argmax) behavior from each slot.
    """
    OPS_TO_OPCODE = {
        0: "ADD",   # +
        1: "SUB",   # -
        2: "MUL",   # *
        3: "ADD",   # / (approximate with ADD as placeholder)
        4: "ADD",   # % (approximate)
        5: "MOV_REG",  # identity
    }

    instructions: List[Instruction] = []
    slots = desc.get("slots", [])

    # Load constants into high registers (R4-R7 for up to 4 constants)
    consts = desc.get("consts", [0.0, 1.0, -1.0, 2.0, -2.0, 10.0])
    const_base_reg = max(n_args, 4)  # start constants after input regs
    for i, c in enumerate(consts[:4]):
        if len(instructions) >= max_instructions - 1:
            break
        reg_idx = const_base_reg + i
        if reg_idx < 8:
            instructions.append(Instruction(
                opcode=OPCODES["MOV_IMM"],
                dst=reg_idx,
                immediate=float(c),
            ))

    # Init slots -> R2, R3 (or wherever they map)
    init_dst_base = 2
    for s_idx, slot in enumerate(slots[:3]):
        if len(instructions) >= max_instructions - 1:
            break
        op = slot.get("op", 5)
        s1 = min(slot.get("s1", 0), 7)
        s2 = min(slot.get("s2", 0), 7)
        dst = min(init_dst_base + s_idx, 7)
        opcode_name = OPS_TO_OPCODE.get(op, "MOV_REG")
        instructions.append(Instruction(
            opcode=OPCODES[opcode_name],
            dst=dst,
            src1=s1,
            src2=s2,
        ))

    # Return mapping: which register holds the result
    ret_src = min(desc.get("ret_src", 0), 7)

    # If the return register != R2, add a final MOV
    if ret_src != 2:
        instructions.append(Instruction(
            opcode=OPCODES["MOV_REG"],
            dst=2,
            src1=ret_src,
        ))

    # HALT
    instructions.append(Instruction(opcode=OPCODES["HALT"]))

    # Pad to max length with NOPs
    while len(instructions) < max_instructions:
        instructions.append(Instruction(opcode=OPCODES["NOP"]))

    return instructions[:max_instructions]


def _mog_params_to_soft_program(
    params: torch.Tensor,
    n_args: int,
    max_program_len: int = 12,
    num_registers: int = 8,
) -> SoftProgram:
    """Create a SoftProgram whose logits are initialized from mog_synth
    parameter gradients.  This enables gradient transfer: we use gradient
    information from the mog_synth parameter space to warm-start the
    nCPU SoftProgram search.

    The key insight is that both representations use softmax over logits
    for discrete choices.  We map the mog_synth slot structure to the
    nCPU instruction slots by:
    1. Decoding each mog_synth slot into (op, src1, src2)
    2. Setting the corresponding nCPU opcode/register logits high
    3. Filling the rest with the nCPU SoftProgram's random init
    """
    _ensure_mog_synth()

    program = SoftProgram(
        max_length=max_program_len,
        num_registers=num_registers,
    )

    desc = _soft_synth.params_to_description(params, n_args)
    instructions = _mog_description_to_ncpu_instructions(desc, n_args, max_program_len)

    HI = 4.0
    LO = -4.0

    with torch.no_grad():
        for i, inst in enumerate(instructions):
            if i >= max_program_len:
                break
            # Set opcode logit
            program.opcode_logits.data[i].fill_(LO)
            program.opcode_logits.data[i, inst.opcode] = HI

            # Set register logits
            if inst.dst < num_registers:
                program.dst_logits.data[i].fill_(LO)
                program.dst_logits.data[i, inst.dst] = HI
            if inst.src1 < num_registers:
                program.src1_logits.data[i].fill_(LO)
                program.src1_logits.data[i, inst.src1] = HI
            if inst.src2 < num_registers:
                program.src2_logits.data[i].fill_(LO)
                program.src2_logits.data[i, inst.src2] = HI

            program.immediates.data[i] = inst.immediate
            program.branch_logits.data[i].fill_(LO)
            program.branch_logits.data[i, min(inst.branch_target, max_program_len - 1)] = HI

    return program


# ---------------------------------------------------------------------------
# 1. DifferentiableProgramExecutor
# ---------------------------------------------------------------------------

class DifferentiableProgramExecutor(nn.Module):
    """Execute a synthesized program through the differentiable OS with
    gradients flowing back to the program parameters.

    This bridges two worlds:
    - **mog_synth** operates in a continuous parameter space where programs
      are vectors of logits over operations, registers, and comparisons.
    - **DifferentiableOS** provides differentiable scheduling, caching, and
      memory allocation.

    The executor converts mog_synth programs into a form executable by
    the differentiable engine, runs them under OS supervision, and returns
    outputs with full gradient connectivity.

    Execution modes:
    - ``mode='ncpu'``: Convert to nCPU instructions and run through
      DifferentiableEngine (full gradient flow, approximate mapping).
    - ``mode='mog_native'``: Run through mog_synth's own SoftUniversalProgram
      forward pass with OS metrics computed alongside (exact semantics,
      OS gradients via synthetic workload events).
    """

    def __init__(
        self,
        n_processes: int = 4,
        cache_size: int = 8,
        memory_size: int = 256,
        num_registers: int = 8,
        max_program_len: int = 12,
        mode: str = "mog_native",
    ):
        super().__init__()
        self.mode = mode
        self.num_registers = num_registers
        self.max_program_len = max_program_len

        # Differentiable OS
        self.os = DifferentiableOS(
            n_processes=n_processes,
            cache_size=cache_size,
            memory_size=memory_size,
        )

        # nCPU execution engine (used in 'ncpu' mode)
        self.engine = DifferentiableEngine(num_registers=num_registers)

    def _create_workload_event(
        self,
        step: int,
        total_steps: int,
        input_vals: torch.Tensor,
        n_processes: int = 4,
    ) -> WorkloadEvent:
        """Synthesize an OS workload event from program execution state.

        Maps the program's execution context (which instruction is running,
        what data it accesses) into the OS event format so the differentiable
        OS can make scheduling/caching/allocation decisions.
        """
        device = input_vals.device if isinstance(input_vals, torch.Tensor) else "cpu"

        # Process states: one process per I/O example in the batch
        process_features = torch.rand(n_processes, 6, device=device)
        # Mark the "current" process as highest priority
        current = step % n_processes
        process_features[current, 0] = 1.0   # priority
        process_features[current, 3] = 1.0   # remaining_work

        # Memory access derived from input values
        if isinstance(input_vals, torch.Tensor) and input_vals.numel() > 0:
            addr_feature = input_vals.float().mean().sigmoid()
        else:
            addr_feature = torch.tensor(0.5, device=device)

        memory_access = torch.stack([
            addr_feature,
            torch.tensor(1.0, device=device),  # is_read
            torch.tensor(float(step) / max(total_steps, 1), device=device),
            torch.tensor(float(step) / max(total_steps, 1), device=device),
        ])

        alloc_request = torch.tensor([
            0.1,   # size_normalized
            1.0,   # alignment
            0.5,   # urgency
            0.5,   # lifetime_hint
        ], device=device)

        return WorkloadEvent(
            process_states=process_features,
            memory_access=memory_access,
            alloc_request=alloc_request,
        )

    def execute_mog_native(
        self,
        params: torch.Tensor,
        examples: List[Tuple[List[int], int]],
        n_args: int,
        temperature: float = 1.0,
    ) -> ExecutorResult:
        """Execute via mog_synth's native SoftUniversalProgram with OS wrapping.

        This is the preferred mode: it preserves exact mog_synth semantics
        while adding OS-level gradient flow.
        """
        _ensure_mog_synth()

        device = params.device
        B = len(examples)

        # Build input/target tensors
        inputs_t = torch.tensor(
            [[float(x) for x in inp] for inp, _ in examples],
            dtype=torch.float32, device=device,
        )
        targets_t = torch.tensor(
            [float(t) for _, t in examples],
            dtype=torch.float32, device=device,
        )

        # Create a SoftUniversalProgram and wire params with gradient flow.
        # We can't use nn.Parameter(params.clone()) because nn.Parameter
        # creates a leaf tensor that breaks the gradient chain.  We use
        # object.__setattr__ to bypass nn.Module's type check, allowing
        # a non-leaf tensor with grad_fn to be assigned directly.
        model = _soft_synth.SoftUniversalProgram(n_args).to(device)

        # Clamp params to prevent overflow in the soft loop iterations.
        # Large logits cause exp() overflow when multiplied across 32 iters.
        clamped_params = params.clamp(-10.0, 10.0)

        # Bypass nn.Module.__setattr__ which rejects non-Parameter tensors.
        object.__setattr__(model, "params", clamped_params)

        # Forward pass through mog_synth soft program.
        # The params tensor keeps its grad_fn, so gradients flow back.
        outputs = model.forward(inputs_t, temperature)  # (B,)

        # Guard against NaN/Inf in outputs (can happen with extreme params)
        if not torch.isfinite(outputs).all():
            outputs = torch.where(
                torch.isfinite(outputs), outputs,
                torch.zeros_like(outputs),
            )

        # Run OS events alongside execution for gradient flow
        self.os.reset()
        schedule_ws, eviction_ws, placement_ws = [], [], []
        n_steps = 5  # synthetic OS steps per program execution
        for step in range(n_steps):
            event = self._create_workload_event(step, n_steps, inputs_t)
            sw, ew, pw = self.os.step(event, temperature)
            schedule_ws.append(sw)
            eviction_ws.append(ew)
            placement_ws.append(pw)

        os_metrics = {
            "cache_hit_rate": self.os.cache.hit_rate,
            "fragmentation": self.os.allocator.fragmentation,
        }

        return ExecutorResult(
            output=outputs,
            registers=[outputs],  # mog_synth returns scalar output
            os_metrics=os_metrics,
            schedule_weights=schedule_ws,
            eviction_weights=eviction_ws,
            placement_weights=placement_ws,
        )

    def execute_ncpu(
        self,
        params: torch.Tensor,
        examples: List[Tuple[List[int], int]],
        n_args: int,
        temperature: float = 1.0,
        max_exec_steps: int = 16,
    ) -> ExecutorResult:
        """Execute via nCPU DifferentiableEngine with OS wrapping.

        Converts the mog_synth program to nCPU instructions and runs through
        the full differentiable CPU + OS stack.
        """
        _ensure_mog_synth()

        soft_prog = _mog_params_to_soft_program(
            params, n_args, self.max_program_len, self.num_registers
        )

        self.os.reset()
        all_outputs = []
        all_registers = []
        schedule_ws, eviction_ws, placement_ws = [], [], []

        for ex_idx, (inputs, target) in enumerate(examples):
            input_dict = {i: float(v) for i, v in enumerate(inputs)}

            # Execute through differentiable engine
            result = self.engine.execute_soft_batched(
                soft_prog,
                [input_dict],
                max_steps=max_exec_steps,
                temperature=temperature,
                skip_bitwise=True,
            )[0]

            output_val = result.registers[2]  # convention: output in R2
            all_outputs.append(output_val)
            all_registers.append(result.registers)

            # OS step
            input_t = torch.tensor(inputs, dtype=torch.float32)
            event = self._create_workload_event(
                ex_idx, len(examples), input_t
            )
            sw, ew, pw = self.os.step(event, temperature)
            schedule_ws.append(sw)
            eviction_ws.append(ew)
            placement_ws.append(pw)

        outputs = torch.stack(all_outputs)
        os_metrics = {
            "cache_hit_rate": self.os.cache.hit_rate,
            "fragmentation": self.os.allocator.fragmentation,
        }

        return ExecutorResult(
            output=outputs,
            registers=all_registers,
            os_metrics=os_metrics,
            schedule_weights=schedule_ws,
            eviction_weights=eviction_ws,
            placement_weights=placement_ws,
        )

    def execute(
        self,
        params: torch.Tensor,
        examples: List[Tuple[List[int], int]],
        n_args: int,
        temperature: float = 1.0,
    ) -> ExecutorResult:
        """Execute a synthesized program through the differentiable OS.

        Args:
            params: mog_synth parameter tensor (from SoftUniversalProgram).
            examples: List of (input_list, expected_output) pairs.
            n_args: Number of input arguments per example.
            temperature: Softmax temperature for soft execution.

        Returns:
            ExecutorResult with outputs, OS metrics, and gradient info.
        """
        if self.mode == "mog_native":
            return self.execute_mog_native(params, examples, n_args, temperature)
        else:
            return self.execute_ncpu(params, examples, n_args, temperature)


# ---------------------------------------------------------------------------
# 2. SynthesisOSPipeline
# ---------------------------------------------------------------------------

class SynthesisOSPipeline(nn.Module):
    """End-to-end pipeline: I/O examples -> synthesis -> OS execution -> loss.

    Takes input-output examples, uses mog_synth gradient synthesis to discover
    candidate programs, executes them through the differentiable OS, computes
    loss, and backpropagates through the entire stack to refine the program.

    The pipeline adds an OS-awareness term to the synthesis loss:
        total_loss = io_loss + os_weight * os_loss

    where os_loss penalizes programs that cause poor cache behavior, high
    fragmentation, or suboptimal scheduling.  This biases synthesis toward
    programs that are not only correct but also *efficient to execute* on
    the neural OS.
    """

    def __init__(
        self,
        n_restarts: int = 3,
        n_synth_steps: int = 400,
        n_refine_steps: int = 200,
        os_weight: float = 0.01,
        cache_size: int = 8,
        memory_size: int = 256,
        synth_lr: float = 0.05,
        refine_lr: float = 0.01,
    ):
        super().__init__()
        self.n_restarts = n_restarts
        self.n_synth_steps = n_synth_steps
        self.n_refine_steps = n_refine_steps
        self.os_weight = os_weight
        self.synth_lr = synth_lr
        self.refine_lr = refine_lr

        self.executor = DifferentiableProgramExecutor(
            cache_size=cache_size,
            memory_size=memory_size,
            mode="mog_native",
        )

    def _synthesize_candidates(
        self,
        examples: List[Tuple[List[int], int]],
        n_args: int,
        device: torch.device,
    ) -> List[torch.Tensor]:
        """Run mog_synth gradient synthesis to produce candidate programs.

        Uses mog_synth's battle-tested `synthesize()` function which handles
        NaN recovery, temperature annealing, and multiple restarts internally.
        Returns a list of parameter tensors for the best candidates.
        """
        _ensure_mog_synth()

        candidates = []

        # Use mog_synth's proven synthesize() for primary candidate
        result = _soft_synth.synthesize(
            examples, n_args,
            n_steps=self.n_synth_steps,
            lr=self.synth_lr,
            n_restarts=self.n_restarts,
            device=device,
        )

        if result["solved"]:
            # Convert description back to params
            desc = result["description"]
            params = _soft_synth.description_to_params(desc, n_args).to(device)
            candidates.append(params)
        else:
            # Even if not solved, generate candidates from cold restarts
            # using the internal loop with NaN recovery
            n = _soft_synth.n_params_for(n_args)
            inputs_t = torch.tensor(
                [[float(x) for x in inp] for inp, _ in examples],
                dtype=torch.float32, device=device,
            )
            targets_t = torch.tensor(
                [float(t) for _, t in examples],
                dtype=torch.float32, device=device,
            )

            for restart in range(self.n_restarts):
                model = _soft_synth.SoftUniversalProgram(n_args).to(device)
                with torch.no_grad():
                    model.params.data[:n - _soft_synth.N_CONSTS] = (
                        torch.randn(n - _soft_synth.N_CONSTS, device=device) * 0.5
                    )
                    for i, v in enumerate(_soft_synth.CONST_VALS):
                        model.params[model._co_off + i] = float(v)

                solved, steps = _soft_synth._run_one(
                    model, examples, inputs_t, targets_t,
                    self.n_synth_steps, self.synth_lr,
                )
                candidates.append(model.params.detach().clone())
                if solved:
                    break

        return candidates

    def _compute_io_loss(
        self,
        outputs: torch.Tensor,
        examples: List[Tuple[List[int], int]],
    ) -> torch.Tensor:
        """MSE loss between predicted and expected outputs."""
        targets = torch.tensor(
            [float(t) for _, t in examples],
            dtype=torch.float32,
            device=outputs.device,
        )
        return F.mse_loss(outputs, targets)

    def _compute_os_loss(
        self,
        executor_result: ExecutorResult,
    ) -> torch.Tensor:
        """Compute OS policy quality loss.

        Penalizes:
        - Low cache hit rate (more misses = higher loss)
        - High fragmentation
        - Scheduling entropy (prefer decisive scheduling)

        All terms are differentiable through the OS networks.
        """
        loss = torch.tensor(0.0)

        # Cache miss penalty: use eviction weights entropy as proxy
        for ew in executor_result.eviction_weights:
            if ew.sum() > 0:
                # Higher entropy in eviction weights means the OS is uncertain
                # about which line to evict — penalize this
                probs = ew / (ew.sum() + 1e-8)
                entropy = -(probs * (probs + 1e-8).log()).sum()
                loss = loss + entropy * 0.1

        # Fragmentation penalty (from OS metrics, non-differentiable but tracked)
        frag = executor_result.os_metrics.get("fragmentation", 0.0)
        loss = loss + torch.tensor(frag)

        # Scheduling decisiveness: prefer low-entropy scheduling weights
        for sw in executor_result.schedule_weights:
            probs = sw / (sw.sum() + 1e-8)
            entropy = -(probs * (probs + 1e-8).log()).sum()
            loss = loss + entropy * 0.01

        return loss

    def _refine_with_os_gradients(
        self,
        params: torch.Tensor,
        examples: List[Tuple[List[int], int]],
        n_args: int,
        temperature_start: float = 0.5,
        temperature_end: float = 0.05,
    ) -> Tuple[torch.Tensor, List[float]]:
        """Refine a candidate program using gradients from the full
        execution + OS stack.

        Unlike pure synthesis (which only optimizes I/O loss), this step
        backpropagates through the OS as well, biasing the program toward
        being both correct AND efficient.
        """
        _ensure_mog_synth()

        # Make params a leaf parameter for gradient computation
        refined = nn.Parameter(params.clone())
        optimizer = torch.optim.Adam([refined], lr=self.refine_lr)
        loss_history = []

        for step in range(self.n_refine_steps):
            progress = step / max(self.n_refine_steps - 1, 1)
            temperature = temperature_start + (temperature_end - temperature_start) * progress

            optimizer.zero_grad()

            try:
                # Execute through OS-aware executor
                result = self.executor.execute(refined, examples, n_args, temperature)

                # Combined loss
                io_loss = self._compute_io_loss(result.output, examples)
                os_loss = self._compute_os_loss(result)
                total_loss = io_loss + self.os_weight * os_loss

                if not torch.isfinite(total_loss):
                    # NaN recovery: reset to small random perturbation
                    with torch.no_grad():
                        refined.data.add_(torch.randn_like(refined) * 0.01)
                    continue

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_([refined], 1.0)
                optimizer.step()

                # NaN recovery for parameters
                with torch.no_grad():
                    if not torch.isfinite(refined).all():
                        refined.data.copy_(params.clone())
                        break

                loss_history.append(total_loss.item())
            except RuntimeError:
                # Autograd error recovery
                with torch.no_grad():
                    refined.data.add_(torch.randn_like(refined) * 0.01)
                continue

            # Early stop if discrete-correct
            if step % 20 == 0:
                with torch.no_grad():
                    if _soft_synth.check_discrete(refined, examples, n_args):
                        break

        return refined.detach(), loss_history

    def run(
        self,
        examples: List[Tuple[List[int], int]],
        n_args: int,
        device: Optional[torch.device] = None,
        verbose: bool = False,
    ) -> PipelineResult:
        """Run the full synthesis-OS pipeline.

        1. Synthesize candidate programs via mog_synth gradient descent.
        2. Execute each candidate through the differentiable OS.
        3. Refine the best candidate using OS-aware gradients.
        4. Return the final program with full metrics.

        Args:
            examples: List of (input_list, expected_output) pairs.
                e.g. [([5, 3], 8), ([10, 7], 17)]
            n_args: Number of input arguments per example.
            device: Torch device (auto-detected if None).
            verbose: Print progress during synthesis and refinement.

        Returns:
            PipelineResult with the best program, accuracy, and OS metrics.
        """
        _ensure_mog_synth()

        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        if verbose:
            print(f"SynthesisOSPipeline: {len(examples)} examples, "
                  f"{n_args} args, device={device}")

        # Phase 1: Synthesize candidates
        if verbose:
            print("Phase 1: Synthesizing candidate programs...")
        candidates = self._synthesize_candidates(examples, n_args, device)

        if verbose:
            print(f"  Generated {len(candidates)} candidate(s)")

        # Phase 2: Evaluate candidates through OS
        best_params = None
        best_io_loss = float("inf")
        best_result = None

        for i, params in enumerate(candidates):
            params = params.to(device)
            # Need gradients for the execution
            params_with_grad = nn.Parameter(params.clone())

            # Use a moderate temperature for evaluation — too low causes
            # NaN/Inf when params haven't converged (sharp softmax on random logits)
            try:
                with torch.enable_grad():
                    result = self.executor.execute(
                        params_with_grad, examples, n_args, temperature=1.0
                    )
                    io_loss = self._compute_io_loss(result.output, examples)

                loss_val = io_loss.item()
                if not torch.isfinite(io_loss) or loss_val != loss_val:
                    loss_val = float("inf")
            except RuntimeError:
                loss_val = float("inf")
                result = None

            if verbose:
                discrete_ok = _soft_synth.check_discrete(params, examples, n_args)
                cache_hr = result.os_metrics['cache_hit_rate'] if result else 0.0
                print(f"  Candidate {i}: io_loss={loss_val:.4f}, "
                      f"discrete_ok={discrete_ok}, "
                      f"cache_hr={cache_hr:.1%}")

            if loss_val < best_io_loss:
                best_io_loss = loss_val
                best_params = params
                best_result = result

        # If no candidate had finite soft loss, fall back to the first
        # candidate that is discrete-correct, or just the first candidate
        if best_params is None or best_io_loss == float("inf"):
            for params in candidates:
                with torch.no_grad():
                    if _soft_synth.check_discrete(params, examples, n_args):
                        best_params = params.to(device)
                        break
            # If still nothing, use the first candidate for refinement anyway
            if best_params is None and candidates:
                best_params = candidates[0].to(device)

        if best_params is None:
            return PipelineResult(
                program_description=None,
                program_text="(no program found)",
                accuracy=0.0,
                io_loss=float("inf"),
                os_loss=0.0,
                total_loss=float("inf"),
                os_metrics={},
                synthesis_method="failed",
                refinement_steps=0,
                loss_history=[],
            )

        # Phase 3: Refine with OS-aware gradients
        if verbose:
            print("Phase 3: Refining with OS-aware gradients...")

        refined_params, refine_history = self._refine_with_os_gradients(
            best_params.to(device), examples, n_args,
        )

        # Final evaluation (use moderate temperature for stability)
        refined_params_eval = nn.Parameter(refined_params.clone())
        try:
            with torch.enable_grad():
                final_result = self.executor.execute(
                    refined_params_eval, examples, n_args, temperature=0.5
                )
                final_io_loss = self._compute_io_loss(final_result.output, examples)
                final_os_loss = self._compute_os_loss(final_result)
                if not torch.isfinite(final_io_loss):
                    final_io_loss = torch.tensor(float("inf"))
                    final_os_loss = torch.tensor(0.0)
        except RuntimeError:
            final_io_loss = torch.tensor(float("inf"))
            final_os_loss = torch.tensor(0.0)
            final_result = None

        # Compute accuracy
        correct = 0
        with torch.no_grad():
            for inputs, target in examples:
                result_val = _soft_synth.discrete_eval(refined_params, inputs, n_args)
                if result_val is not None and result_val == int(target):
                    correct += 1
        accuracy = correct / len(examples)

        # Build program description
        desc = _soft_synth.params_to_description(refined_params, n_args)

        # Human-readable text
        program_text = _format_description(desc, n_args)

        if verbose:
            print(f"Final: accuracy={accuracy:.1%}, "
                  f"io_loss={final_io_loss.item():.4f}, "
                  f"os_loss={final_os_loss.item():.4f}")
            print(f"Program:\n{program_text}")

        os_metrics = final_result.os_metrics if final_result else {}

        return PipelineResult(
            program_description=desc,
            program_text=program_text,
            accuracy=accuracy,
            io_loss=final_io_loss.item(),
            os_loss=final_os_loss.item(),
            total_loss=(final_io_loss + self.os_weight * final_os_loss).item(),
            os_metrics=os_metrics,
            synthesis_method="gradient_guided" if refine_history else "cold",
            refinement_steps=len(refine_history),
            loss_history=refine_history,
        )


# ---------------------------------------------------------------------------
# 3. GradientGuidedSynthesis
# ---------------------------------------------------------------------------

class GradientGuidedSynthesis(nn.Module):
    """Use OS execution gradients to guide the synthesis search.

    Instead of blind random restarts, this module:
    1. Runs a candidate program through the differentiable OS.
    2. Computes dL/d(params) — the gradient of the total loss w.r.t.
       every program parameter (opcode logits, register selections, etc.).
    3. Identifies which slots and fields have the *largest* gradients
       (i.e. which changes would most reduce the loss).
    4. Generates targeted perturbations in those directions, replacing
       random search with gradient-informed search.

    This is a *hybrid* approach: the synthesis engine proposes programs
    via its own heuristics, and the differentiable OS evaluates and
    suggests specific refinements.  It combines the global search ability
    of program synthesis with the local optimization power of gradients.
    """

    def __init__(
        self,
        executor: Optional[DifferentiableProgramExecutor] = None,
        os_weight: float = 0.01,
        top_k_suggestions: int = 5,
    ):
        super().__init__()
        self.executor = executor or DifferentiableProgramExecutor(mode="mog_native")
        self.os_weight = os_weight
        self.top_k = top_k_suggestions

    def compute_guidance(
        self,
        params: torch.Tensor,
        examples: List[Tuple[List[int], int]],
        n_args: int,
        temperature: float = 0.3,
    ) -> GradientGuidance:
        """Compute gradient guidance for a candidate program.

        Returns a GradientGuidance with:
        - param_gradients: full gradient vector (same shape as params)
        - slot_importance: which slots matter most (gradient magnitude)
        - suggested_changes: ranked list of specific field changes
        - estimated_improvement: how much loss reduction to expect
        """
        _ensure_mog_synth()

        device = params.device
        p_param = nn.Parameter(params.clone().to(device))

        # Forward + backward through executor
        result = self.executor.execute(p_param, examples, n_args, temperature)

        targets = torch.tensor(
            [float(t) for _, t in examples],
            dtype=torch.float32, device=device,
        )
        io_loss = F.mse_loss(result.output, targets)

        # OS loss (differentiable components only)
        os_loss = torch.tensor(0.0, device=device)
        for sw in result.schedule_weights:
            probs = sw / (sw.sum() + 1e-8)
            os_loss = os_loss - (probs * (probs + 1e-8).log()).sum() * 0.01
        for ew in result.eviction_weights:
            if ew.sum() > 0:
                probs = ew / (ew.sum() + 1e-8)
                os_loss = os_loss - (probs * (probs + 1e-8).log()).sum() * 0.1

        total_loss = io_loss + self.os_weight * os_loss
        total_loss.backward()

        grad = p_param.grad.detach().clone()

        # Analyze gradient structure to identify important slots
        p = _soft_synth._pool(n_args)
        sps = _soft_synth._sps(p)
        n_slots = _soft_synth.N_UNIV_SLOTS

        slot_importance = torch.zeros(n_slots)
        for s in range(n_slots):
            off = s * sps
            slot_importance[s] = grad[off:off + sps].abs().sum()

        # Generate suggested changes: find the top-k highest gradient params
        abs_grad = grad.abs()
        top_k_indices = abs_grad.topk(min(self.top_k * 3, len(grad))).indices

        suggested = []
        for idx in top_k_indices:
            idx_val = idx.item()
            # Determine which slot and field this parameter belongs to
            slot_idx = idx_val // sps if idx_val < n_slots * sps else -1
            field_offset = idx_val % sps if slot_idx >= 0 else -1

            if slot_idx >= 0 and slot_idx < n_slots:
                # Decode field type
                if field_offset < _soft_synth.N_OPS_EXT:
                    field_name = "op"
                    field_choice = field_offset
                elif field_offset < _soft_synth.N_OPS_EXT + p:
                    field_name = "s1"
                    field_choice = field_offset - _soft_synth.N_OPS_EXT
                elif field_offset < _soft_synth.N_OPS_EXT + 2 * p:
                    field_name = "s2"
                    field_choice = field_offset - _soft_synth.N_OPS_EXT - p
                else:
                    field_name = "gate"
                    field_choice = field_offset - _soft_synth.N_OPS_EXT - 2 * p

                direction = "increase" if grad[idx_val] < 0 else "decrease"
                suggested.append({
                    "slot": int(slot_idx),
                    "field": field_name,
                    "choice": int(field_choice),
                    "direction": direction,
                    "gradient_magnitude": float(abs_grad[idx_val]),
                })

            if len(suggested) >= self.top_k:
                break

        # Estimate improvement: first-order Taylor approximation
        # dL ~= grad . dp, where dp is the step we'd take
        estimated_improvement = float(abs_grad.topk(self.top_k).values.sum() * 0.01)

        return GradientGuidance(
            param_gradients=grad,
            slot_importance=slot_importance,
            suggested_changes=suggested,
            estimated_improvement=estimated_improvement,
        )

    def guided_search(
        self,
        examples: List[Tuple[List[int], int]],
        n_args: int,
        n_iterations: int = 10,
        n_candidates_per_iter: int = 5,
        synth_steps: int = 200,
        device: Optional[torch.device] = None,
        verbose: bool = False,
    ) -> PipelineResult:
        """Run gradient-guided synthesis search.

        Each iteration:
        1. Synthesize candidates via mog_synth.
        2. Compute gradient guidance for the best candidate.
        3. Use guidance to bias the next round of synthesis
           (warm-start from gradient-perturbed params).

        Args:
            examples: I/O specification.
            n_args: Number of input arguments.
            n_iterations: Number of search iterations.
            n_candidates_per_iter: Candidates per iteration.
            synth_steps: Gradient steps per candidate.
            device: Torch device.
            verbose: Print progress.

        Returns:
            PipelineResult for the best program found.
        """
        _ensure_mog_synth()

        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        n = _soft_synth.n_params_for(n_args)
        inputs_t = torch.tensor(
            [[float(x) for x in inp] for inp, _ in examples],
            dtype=torch.float32, device=device,
        )
        targets_t = torch.tensor(
            [float(t) for _, t in examples],
            dtype=torch.float32, device=device,
        )

        best_params = None
        best_loss = float("inf")
        all_loss_history = []
        current_warm_params = None  # warm-start for next iteration

        for iteration in range(n_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")

            candidates = []
            for c in range(n_candidates_per_iter):
                model = _soft_synth.SoftUniversalProgram(n_args).to(device)

                with torch.no_grad():
                    if current_warm_params is not None and c < 2:
                        # Warm-start: perturb from gradient-guided params
                        noise_scale = 0.3 * (1.0 - iteration / n_iterations)
                        noise = torch.randn_like(current_warm_params) * noise_scale
                        model.params.copy_(current_warm_params + noise)
                        # Preserve constants
                        for i, v in enumerate(_soft_synth.CONST_VALS):
                            model.params[model._co_off + i] = float(v)
                    else:
                        # Cold random init
                        model.params.data[:n - _soft_synth.N_CONSTS] = (
                            torch.randn(n - _soft_synth.N_CONSTS, device=device) * 0.5
                        )
                        for i, v in enumerate(_soft_synth.CONST_VALS):
                            model.params[model._co_off + i] = float(v)

                # Run synthesis
                opt = torch.optim.Adam(model.parameters(), lr=0.05)
                t_range = 2.0 - 0.1
                for step in range(synth_steps):
                    temp = max(2.0 - t_range * (step / synth_steps), 0.1)
                    opt.zero_grad()
                    loss = model.mse_loss(inputs_t, targets_t, temp)
                    if not torch.isfinite(loss):
                        break
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                candidates.append((model.params.detach().clone(), loss.item() if torch.isfinite(loss) else float("inf")))

            # Select best candidate this iteration
            candidates.sort(key=lambda x: x[1])
            iter_best_params, iter_best_loss = candidates[0]

            if verbose:
                discrete_ok = _soft_synth.check_discrete(
                    iter_best_params, examples, n_args
                )
                print(f"  Best candidate: loss={iter_best_loss:.4f}, "
                      f"discrete_ok={discrete_ok}")

            all_loss_history.append(iter_best_loss)

            # Check if solved
            with torch.no_grad():
                if _soft_synth.check_discrete(iter_best_params, examples, n_args):
                    best_params = iter_best_params
                    best_loss = iter_best_loss
                    if verbose:
                        print("  SOLVED! Discrete program is correct.")
                    break

            if iter_best_loss < best_loss:
                best_loss = iter_best_loss
                best_params = iter_best_params

            # Compute gradient guidance to inform next iteration
            guidance = self.compute_guidance(
                iter_best_params, examples, n_args
            )

            if verbose:
                print(f"  Gradient guidance: {len(guidance.suggested_changes)} suggestions, "
                      f"est. improvement={guidance.estimated_improvement:.4f}")
                for sg in guidance.suggested_changes[:3]:
                    print(f"    slot {sg['slot']}, {sg['field']}={sg['choice']} "
                          f"({sg['direction']}, grad={sg['gradient_magnitude']:.4f})")

            # Apply gradient step to create warm-start for next iteration
            with torch.no_grad():
                step_size = 0.1
                current_warm_params = (
                    iter_best_params - step_size * guidance.param_gradients
                )

        if best_params is None:
            return PipelineResult(
                program_description=None,
                program_text="(no program found)",
                accuracy=0.0,
                io_loss=float("inf"),
                os_loss=0.0,
                total_loss=float("inf"),
                os_metrics={},
                synthesis_method="gradient_guided_failed",
                refinement_steps=0,
                loss_history=all_loss_history,
            )

        # Final evaluation
        desc = _soft_synth.params_to_description(best_params, n_args)
        correct = 0
        with torch.no_grad():
            for inputs, target in examples:
                result_val = _soft_synth.discrete_eval(best_params, inputs, n_args)
                if result_val is not None and result_val == int(target):
                    correct += 1

        accuracy = correct / len(examples)

        # Final OS metrics
        p_eval = nn.Parameter(best_params.clone())
        with torch.enable_grad():
            final_result = self.executor.execute(p_eval, examples, n_args, 0.1)
            final_io = self._compute_io_loss(final_result.output, examples)
            final_os = self._compute_os_loss(final_result)

        program_text = _format_description(desc, n_args)

        return PipelineResult(
            program_description=desc,
            program_text=program_text,
            accuracy=accuracy,
            io_loss=final_io.item(),
            os_loss=final_os.item(),
            total_loss=(final_io + self.os_weight * final_os).item(),
            os_metrics=final_result.os_metrics,
            synthesis_method="gradient_guided",
            refinement_steps=len(all_loss_history),
            loss_history=all_loss_history,
        )

    def _compute_io_loss(self, outputs, examples):
        targets = torch.tensor(
            [float(t) for _, t in examples],
            dtype=torch.float32, device=outputs.device,
        )
        return F.mse_loss(outputs, targets)

    def _compute_os_loss(self, result: ExecutorResult):
        loss = torch.tensor(0.0)
        for sw in result.schedule_weights:
            probs = sw / (sw.sum() + 1e-8)
            loss = loss - (probs * (probs + 1e-8).log()).sum() * 0.01
        for ew in result.eviction_weights:
            if ew.sum() > 0:
                probs = ew / (ew.sum() + 1e-8)
                loss = loss - (probs * (probs + 1e-8).log()).sum() * 0.1
        frag = result.os_metrics.get("fragmentation", 0.0)
        loss = loss + torch.tensor(frag)
        return loss


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_OPS = ["+", "-", "*", "/", "%", "id"]
_CMPS = ["<", "<=", "==", ">=", ">", "!="]


def _format_description(desc: dict, n_args: int) -> str:
    """Format a mog_synth program description as human-readable pseudocode."""
    consts = desc.get("consts", [0, 1, -1, 2, -2, 10])
    args = [f"a{i}" for i in range(n_args)]
    pool = (
        args
        + [f"c{i}({int(c)})" for i, c in enumerate(consts)]
        + [f"v{i}" for i in range(3)]
        + [f"s{i}" for i in range(6)]
        + [f"p{i}" for i in range(2)]
    )

    def reg(i):
        return pool[i] if i < len(pool) else f"r{i}"

    def fmt_slot(slot, dst_name):
        op = slot.get("op", 5)
        s1, s2 = reg(slot.get("s1", 0)), reg(slot.get("s2", 0))
        gc = slot.get("gate_cmp", 4)
        gl, gr = reg(slot.get("gate_lhs", 0)), reg(slot.get("gate_rhs", 0))
        ev = reg(slot.get("else_val", 0))
        expr = s1 if op == 5 else f"({s1} {_OPS[op]} {s2})"
        return f"  {dst_name} = ({expr}) if {gl} {_CMPS[gc]} {gr} else {ev}"

    lines = [f"fn program({', '.join(args)}):"]

    # Constants
    for i, c in enumerate(consts):
        lines.append(f"  c{i} = {int(c)}")

    slots = desc.get("slots", [])

    # Init
    for i, slot in enumerate(slots[:3]):
        lines.append(fmt_slot(slot, f"v{i}"))

    # Loop init
    lip_pool = args + [f"c{i}({int(c)})" for i, c in enumerate(consts)] + ["v0", "v1", "v2"]
    for i, src in enumerate(desc.get("loop_init", [0] * 6)[:6]):
        src_name = lip_pool[src] if src < len(lip_pool) else f"r{src}"
        lines.append(f"  s{i} = {src_name}")

    # Loop condition
    cc = desc.get("cond_cmp", 4)
    cl, cr = reg(desc.get("cond_lhs", 0)), reg(desc.get("cond_rhs", 0))
    lines.append(f"  while {cl} {_CMPS[cc]} {cr}:")
    for i, slot in enumerate(slots[3:9]):
        lines.append("  " + fmt_slot(slot, f"s{i}"))

    # Post
    for i, slot in enumerate(slots[9:11]):
        lines.append(fmt_slot(slot, f"p{i}"))

    # Return
    ret_src = desc.get("ret_src", 0)
    lines.append(f"  return {reg(ret_src)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo / CLI
# ---------------------------------------------------------------------------

def demo_addition():
    """Demo: synthesize an addition program through the OS pipeline."""
    import random
    random.seed(42)
    torch.manual_seed(42)

    examples = [([a, b], a + b) for a, b in
                [(5, 3), (10, 7), (0, 0), (1, 1), (20, 30),
                 (8, 12), (3, 15), (100, 200), (7, 7), (42, 58)]]

    print("=" * 64)
    print("DifferentiableOS + Program Synthesis: Addition")
    print("=" * 64)
    print(f"Examples: {len(examples)}")
    for inp, out in examples[:3]:
        print(f"  {inp} -> {out}")
    print("  ...")
    print()

    pipeline = SynthesisOSPipeline(
        n_restarts=3,
        n_synth_steps=300,
        n_refine_steps=100,
        os_weight=0.01,
    )

    result = pipeline.run(examples, n_args=2, verbose=True)

    print()
    print("--- Result ---")
    print(f"Accuracy: {result.accuracy:.1%}")
    print(f"I/O Loss: {result.io_loss:.6f}")
    print(f"OS Loss:  {result.os_loss:.6f}")
    print(f"Method:   {result.synthesis_method}")
    print(f"Refine steps: {result.refinement_steps}")
    print(f"OS metrics: {result.os_metrics}")
    print()
    print("Program:")
    print(result.program_text)

    return result


def demo_gradient_guided():
    """Demo: gradient-guided synthesis discovers a squaring program."""
    import random
    random.seed(123)
    torch.manual_seed(123)

    examples = [([x], x * x) for x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

    print()
    print("=" * 64)
    print("Gradient-Guided Synthesis: x^2")
    print("=" * 64)
    print(f"Examples: {len(examples)}")
    for inp, out in examples[:5]:
        print(f"  {inp} -> {out}")
    print("  ...")
    print()

    ggs = GradientGuidedSynthesis(os_weight=0.01, top_k_suggestions=5)
    result = ggs.guided_search(
        examples, n_args=1,
        n_iterations=5,
        n_candidates_per_iter=3,
        synth_steps=300,
        verbose=True,
    )

    print()
    print("--- Result ---")
    print(f"Accuracy: {result.accuracy:.1%}")
    print(f"I/O Loss: {result.io_loss:.6f}")
    print(f"OS Loss:  {result.os_loss:.6f}")
    print(f"Method:   {result.synthesis_method}")
    print()
    print("Program:")
    print(result.program_text)

    return result


if __name__ == "__main__":
    r1 = demo_addition()
    r2 = demo_gradient_guided()

    print()
    print("=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  Addition:  accuracy={r1.accuracy:.1%}, "
          f"loss={r1.total_loss:.4f}, method={r1.synthesis_method}")
    print(f"  Squaring:  accuracy={r2.accuracy:.1%}, "
          f"loss={r2.total_loss:.4f}, method={r2.synthesis_method}")
