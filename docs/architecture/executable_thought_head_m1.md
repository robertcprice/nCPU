# Executable Thought Head (M1)

## Goal

Close the first end-to-end loop for Neural-Physical Chain of Thought:

1. Start from a transformer hidden state `h_t`.
2. Decode `h_t` into an executable differentiable program.
3. Run that program on the differentiable nCPU execution engine.
4. Summarize the execution trace.
5. Feed the trace back into the hidden state as a learned patch.

This milestone is about proving the gradient path works end to end. It is not
yet the full thesis artifact.

## Repo Reality

The repo already has the three pieces needed for an M1 closure:

- `ncpu.differentiable.diff_compiler.DifferentiableCompiler`
  - already exposes `decode(context)` and can emit `SoftProgram` directly from a
    continuous vector.
- `ncpu.differentiable.execution.DifferentiableEngine`
  - already executes `SoftProgram` with gradient flow through opcodes,
    registers, immediates, and branches.
- `ncpu.self_optimizing.latent_heads.state_patch_head.StatePatchHead`
  - already turns a compact execution summary into a learned patch signal.

What the repo does not have yet is a native latent-to-`MogDiffCompiler.compile`
path. The current `MogDiffCompiler` is example-driven. For M1, the correct
reuse point is the differentiable compiler decoder path, not the example-driven
compiler entry point.

## Implemented Contract

New module:

- `ncpu/self_optimizing/executable_thought_head.py`

Primary forward path:

```text
hidden_state
  -> context_projector
  -> DifferentiableCompiler.decode(context)
  -> SoftProgram
  -> DifferentiableEngine.execute_soft(...)
  -> trace pooling
  -> StatePatchHead
  -> hidden_state + delta
```

Inputs:

- `hidden_state`: `[hidden_dim]` or `[batch, hidden_dim]`
- `register_inputs`: `[num_registers]` or `[batch, num_registers]`

Outputs:

- compiler context vector
- predicted scalar output from a configured output register
- next hidden state after a learned patch
- pooled trace projection
- raw patch signal
- final register state and flags
- rendered instruction text
- Mog-style preview text for inspectability

## Structural Priors

M1 uses a deliberately narrow structural prior so a smoke run can converge in a
single session:

- arithmetic-only opcode mask by default:
  - `NOP`, `MOV_IMM`, `MOV_REG`, `ADD`, `SUB`, `MUL`, `HALT`
- register priors bias the first active slot toward:
  - `src1 = R0`
  - `src2 = R1`
  - `dst = output_register`
- later slots are biased toward `HALT` / `NOP`

This keeps the artifact honest:

- hidden state still decodes through the real compiler decoder
- execution still happens on the real differentiable nCPU engine
- the search space is just narrowed enough to make the first closure tractable

## Why This Shape

This M1 artifact optimizes for three things:

1. **Real hidden-state decoding**
   - the compiler now directly consumes latent context rather than token IDs
2. **Real executable thoughts**
   - the emitted object is an actual `SoftProgram`, not a side-channel vector
3. **Real feedback into state**
   - execution trace statistics are turned into a learned patch on the hidden
     state

## Inspectability

The module renders both:

- assembly-style `SoftProgram.format_program()`
- a small Mog-style preview for the dominant discrete program

That preview is intentionally a preview, not a claim that the native
`MogDiffCompiler` is already latent-conditioned. The next step is to add a
native Mog IR decoder beside the nCPU decoder and compare them on the same
hidden-state signal.

## Smoke Training Task

The included smoke helper is intentionally small:

- hidden state encodes a tiny operation family
- register inputs carry operand values
- loss is MSE on the configured output register

Success criterion:

- loss drops materially during a short run
- the final rendered program is executable and legible

This proves the core claim for M1:

> a latent state can be decoded into an executable differentiable program whose
> execution trace feeds back into the latent state through backpropagation

## Deferred to M2 / M3

Not in this change:

- latent-conditioned native `MogDiffCompiler.compile`
- array-valued thought programs
- program library persistence
- hard execution fallback on the Metal runtime
- controller-runtime integration with halt policy and memory policy

Those depend on the M1 closure existing first.
