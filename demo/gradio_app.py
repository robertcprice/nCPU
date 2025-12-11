"""KVRM-CPU Interactive Demo.

A Gradio web interface for running and visualizing KVRM-CPU execution.

Usage:
    cd /path/to/kvrm-cpu
    python demo/gradio_app.py

Features:
    - Write or load assembly programs
    - Choose between mock and real (LLM) decoder modes
    - See step-by-step execution trace
    - Visualize register state changes
    - Full decode key auditability
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import gradio as gr
from kvrm_cpu import KVRMCPU


# =============================================================================
# Example Programs
# =============================================================================

EXAMPLE_PROGRAMS = {
    "Sum 1-10": """    MOV R0, 0       ; sum = 0
    MOV R1, 1       ; counter = 1
    MOV R2, 11      ; limit
    MOV R3, 1       ; increment
loop:
    ADD R0, R0, R1  ; sum += counter
    ADD R1, R1, R3  ; counter++
    CMP R1, R2      ; compare to limit
    JNZ loop        ; continue if not equal
    HALT            ; R0 = 55""",

    "Fibonacci(10)": """    MOV R0, 0       ; fib(0) - previous
    MOV R1, 1       ; fib(1) - current
    MOV R2, 10      ; N iterations
    MOV R3, 0       ; counter
    MOV R4, 1       ; constant 1
loop:
    MOV R5, R1      ; temp = current
    ADD R1, R0, R1  ; current = prev + current
    MOV R0, R5      ; prev = temp
    ADD R3, R3, R4  ; counter++
    CMP R3, R2
    JNZ loop
    HALT            ; R1 = 89""",

    "Multiply 7x6": """    MOV R0, 0       ; result = 0
    MOV R1, 7       ; multiplicand
    MOV R2, 6       ; multiplier
    MOV R3, 1       ; decrement constant
    MOV R4, 0       ; zero for comparison
loop:
    ADD R0, R0, R1  ; result += multiplicand
    SUB R2, R2, R3  ; multiplier--
    CMP R2, R4      ; compare to zero
    JNZ loop
    HALT            ; R0 = 42""",

    "Simple Mov": """    MOV R0, 42
    MOV R1, R0
    MOV R2, 100
    HALT""",

    "Custom": ""
}


# =============================================================================
# Execution Functions
# =============================================================================

def run_program(program: str, mode: str, model_path: str, max_cycles: int) -> tuple:
    """Execute an assembly program and return results.

    Args:
        program: Assembly source code
        mode: 'mock' or 'real'
        model_path: Path to trained model (for real mode)
        max_cycles: Maximum execution cycles

    Returns:
        Tuple of (summary_text, trace_text, registers_text)
    """
    if not program.strip():
        return "Error: No program provided", "", ""

    try:
        # Initialize CPU
        mock_mode = mode == "mock"
        model = model_path if not mock_mode else None

        cpu = KVRMCPU(
            mock_mode=mock_mode,
            model_path=model,
            max_cycles=max_cycles
        )

        # Load model for real mode
        if not mock_mode:
            if not model_path or not Path(model_path).exists():
                return f"Error: Model not found at {model_path}", "", ""
            cpu.load()

        # Load and run program
        cpu.load_program(program)

        try:
            trace = cpu.run()
        except RuntimeError as e:
            error_msg = str(e)
            trace = cpu.get_trace()
        else:
            error_msg = None

        # Format summary
        summary = cpu.get_summary()
        summary_lines = [
            "EXECUTION SUMMARY",
            "=" * 40,
            f"Cycles: {summary['cycles']}",
            f"Halted: {'Yes' if summary['halted'] else 'No'}",
            f"Errors: {len(summary['errors'])}",
        ]
        if error_msg:
            summary_lines.append(f"\nRuntime: {error_msg}")
        if summary['errors']:
            summary_lines.append("\nDecode Errors:")
            for err in summary['errors'][:5]:
                summary_lines.append(f"  - {err}")

        summary_text = "\n".join(summary_lines)

        # Format trace
        trace_lines = [
            "EXECUTION TRACE",
            "=" * 60,
        ]
        for entry in trace[:100]:  # Limit to 100 entries
            trace_lines.append(f"\n--- Cycle {entry.cycle} (PC={entry.pre_state['pc']}) ---")
            trace_lines.append(f"Instruction: {entry.instruction}")
            trace_lines.append(f"Decoded Key: {entry.key}")
            trace_lines.append(f"Parameters:  {entry.params}")

            # Show register changes
            pre_regs = entry.pre_state['registers']
            post_regs = entry.post_state['registers']
            changes = []
            for reg in sorted(pre_regs.keys()):
                if pre_regs[reg] != post_regs[reg]:
                    changes.append(f"{reg}: {pre_regs[reg]} -> {post_regs[reg]}")
            if changes:
                trace_lines.append(f"Changes:     {', '.join(changes)}")

        if len(trace) > 100:
            trace_lines.append(f"\n... ({len(trace) - 100} more entries)")

        trace_text = "\n".join(trace_lines)

        # Format registers
        regs = cpu.dump_registers()
        flags = summary['flags']
        reg_lines = [
            "FINAL REGISTERS",
            "=" * 30,
        ]
        for reg in sorted(regs.keys()):
            marker = " *" if regs[reg] != 0 else ""
            reg_lines.append(f"  {reg}: {regs[reg]:>10}{marker}")

        reg_lines.append("")
        reg_lines.append("FLAGS")
        reg_lines.append("-" * 30)
        for flag, value in flags.items():
            reg_lines.append(f"  {flag}: {value}")

        registers_text = "\n".join(reg_lines)

        # Cleanup
        if not mock_mode:
            cpu.unload()

        return summary_text, trace_text, registers_text

    except Exception as e:
        return f"Error: {str(e)}", "", ""


def load_example(example_name: str) -> str:
    """Load an example program."""
    return EXAMPLE_PROGRAMS.get(example_name, "")


# =============================================================================
# Gradio Interface
# =============================================================================

def create_demo():
    """Create and return the Gradio demo interface."""

    with gr.Blocks(title="KVRM-CPU Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # KVRM-CPU: Model-Native CPU Emulator

        A CPU emulator where instruction decoding is performed by a semantic LLM,
        emitting verified registry keys that map to pure execution primitives.

        **Architecture**: `fetch -> decode_llm -> key -> verified_execute -> state`

        This demonstrates the KVRM paradigm at the lowest level of computing.
        """)

        with gr.Row():
            with gr.Column(scale=2):
                # Program input
                gr.Markdown("### Assembly Program")

                example_dropdown = gr.Dropdown(
                    choices=list(EXAMPLE_PROGRAMS.keys()),
                    value="Sum 1-10",
                    label="Load Example"
                )

                program_input = gr.Textbox(
                    value=EXAMPLE_PROGRAMS["Sum 1-10"],
                    label="Source Code",
                    lines=15,
                    placeholder="Enter assembly code here..."
                )

                # Settings
                gr.Markdown("### Settings")

                with gr.Row():
                    mode_radio = gr.Radio(
                        choices=["mock", "real"],
                        value="mock",
                        label="Decoder Mode",
                        info="Mock: rule-based | Real: trained LLM"
                    )
                    max_cycles = gr.Slider(
                        minimum=100,
                        maximum=100000,
                        value=10000,
                        step=100,
                        label="Max Cycles"
                    )

                model_path = gr.Textbox(
                    value="models/decode_llm",
                    label="Model Path (for real mode)",
                    visible=True
                )

                run_button = gr.Button("Run Program", variant="primary")

            with gr.Column(scale=3):
                # Results
                with gr.Row():
                    summary_output = gr.Textbox(
                        label="Summary",
                        lines=10,
                        interactive=False
                    )
                    registers_output = gr.Textbox(
                        label="Final Registers",
                        lines=10,
                        interactive=False
                    )

                trace_output = gr.Textbox(
                    label="Execution Trace",
                    lines=20,
                    interactive=False
                )

        # ISA Reference
        with gr.Accordion("ISA Reference", open=False):
            gr.Markdown("""
            | Instruction | Description | Example |
            |-------------|-------------|---------|
            | `MOV Rd, imm` | Load immediate value | `MOV R0, 42` |
            | `MOV Rd, Rs` | Copy register | `MOV R1, R0` |
            | `ADD Rd, Rs1, Rs2` | Add registers | `ADD R3, R1, R2` |
            | `SUB Rd, Rs1, Rs2` | Subtract registers | `SUB R3, R1, R2` |
            | `MUL Rd, Rs1, Rs2` | Multiply registers | `MUL R3, R1, R2` |
            | `CMP Rs1, Rs2` | Compare (sets flags) | `CMP R1, R2` |
            | `JMP addr/label` | Unconditional jump | `JMP loop` |
            | `JZ addr/label` | Jump if zero | `JZ done` |
            | `JNZ addr/label` | Jump if not zero | `JNZ loop` |
            | `HALT` | Stop execution | `HALT` |
            | `NOP` | No operation | `NOP` |

            **Registers**: R0-R7 (8 general purpose, 32-bit signed)
            **Flags**: ZF (zero flag), SF (sign flag)
            **Labels**: Use `name:` to define, reference by name in jumps
            """)

        # KVRM Architecture
        with gr.Accordion("KVRM Architecture", open=False):
            gr.Markdown("""
            ### Traditional CPU vs KVRM-CPU

            **Traditional**: `fetch -> decode -> execute` (hardcoded silicon logic)

            **KVRM-CPU**: `fetch -> decode_llm -> key -> verified_execute`

            ### Key Differences

            1. **Semantic Decode**: Instructions decoded by meaning, not bit patterns
            2. **Verified Registry**: Only pre-defined operation keys can execute
            3. **Full Auditability**: Every decode decision is traceable
            4. **Extensibility**: New instructions via training, not silicon

            ### Registry Keys (12 primitives)

            | Key | Operation |
            |-----|-----------|
            | `OP_MOV_REG_IMM` | Load immediate to register |
            | `OP_MOV_REG_REG` | Register-to-register copy |
            | `OP_ADD` | Addition |
            | `OP_SUB` | Subtraction |
            | `OP_MUL` | Multiplication |
            | `OP_CMP` | Comparison (sets flags) |
            | `OP_JMP` | Unconditional jump |
            | `OP_JZ` | Jump if zero |
            | `OP_JNZ` | Jump if not zero |
            | `OP_HALT` | Stop execution |
            | `OP_NOP` | No operation |
            | `OP_INVALID` | Error handling |
            """)

        # Event handlers
        example_dropdown.change(
            fn=load_example,
            inputs=[example_dropdown],
            outputs=[program_input]
        )

        run_button.click(
            fn=run_program,
            inputs=[program_input, mode_radio, model_path, max_cycles],
            outputs=[summary_output, trace_output, registers_output]
        )

        # Show/hide model path based on mode
        def toggle_model_path(mode):
            return gr.update(visible=(mode == "real"))

        mode_radio.change(
            fn=toggle_model_path,
            inputs=[mode_radio],
            outputs=[model_path]
        )

    return demo


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7861,
        show_error=True
    )
