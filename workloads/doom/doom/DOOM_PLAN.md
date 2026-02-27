# 🎮 DOOM on NeuralOS - Proof of Concept Plan

## Goal
Demonstrate that the KVRM NeuralOS can execute real software by running DOOM.

## Approaches (Ranked by Feasibility)

### Option A: doomgeneric (Recommended)
**doomgeneric** is a minimal, portable DOOM implementation that abstracts all I/O.

```
doomgeneric/
├── doomgeneric.c      # Main loop, calls our DG_* functions
├── doomgeneric.h      # Interface we implement
└── *.c                # DOOM game logic
```

**We implement:**
- `DG_Init()` - Initialize display
- `DG_DrawFrame(pixels)` - Render frame to screen
- `DG_GetKey()` - Get keyboard input
- `DG_SleepMs()` - Timing

**Challenge:** DOOM is written in C, we need to either:
1. Compile to our instruction set (complex)
2. Interpret C at runtime (very slow)
3. Use existing ARM64/x86 DOOM + emulate instructions (feasible!)

### Option B: Instruction-Level Emulation (Most Realistic)
Run pre-compiled DOOM binary, emulate each instruction with Neural CPU.

```python
# doom_emulator.py
def run_doom():
    cpu = NeuralOS()
    memory = [0] * (64 * 1024 * 1024)  # 64MB

    # Load DOOM binary
    load_elf("doom.elf", memory)

    while True:
        # Fetch instruction
        instruction = memory[cpu.pc]

        # Decode (using our trained decoder or classical)
        opcode, rd, rn, rm, imm = decode(instruction)

        # Execute on Neural ALU
        result = cpu.execute(opcode, rd, rn, rm, imm)

        # Handle I/O syscalls
        if is_syscall(instruction):
            handle_io(cpu, memory)
```

### Option C: Neural Renderer Demo (Simplest)
Just demonstrate the Neural CPU can do DOOM-like graphics math.

```python
# doom_render_demo.py
def render_column(wall_height, texture_col):
    """Render one vertical column - DOOM's core rendering."""
    for y in range(SCREEN_HEIGHT):
        # Calculate texture coordinate (requires MUL, DIV)
        tex_y = cpu.execute(MUL, y, wall_height)
        tex_y = cpu.execute(DIV, tex_y, SCREEN_HEIGHT)

        # Get pixel color (requires memory read + AND for masking)
        color = cpu.memory_read(texture_base + tex_col * 64 + tex_y)
        color = cpu.execute(AND, color, 0xFF)

        framebuffer[x * SCREEN_HEIGHT + y] = color
```

## Implementation Plan

### Phase 1: Neural Renderer Demo (This Week)
1. Implement basic raycasting math using Neural CPU
2. Render a simple DOOM-like scene
3. Prove ADD, SUB, MUL, shifts work for graphics

### Phase 2: Simple Game Loop (Next)
1. Load a DOOM WAD file
2. Parse map geometry
3. Render first-person view using Neural CPU
4. Basic movement (WASD)

### Phase 3: Full DOOM (Stretch Goal)
1. ARM64 instruction emulator using Neural CPU
2. Load compiled DOOM ELF
3. Full gameplay

## Required Neural Operations for DOOM Rendering

| Operation | DOOM Use Case | Current FusedALU |
|-----------|---------------|------------------|
| ADD | Coordinate math | ✅ 100% |
| SUB | Distance calc | ✅ 99% |
| MUL | Texture scaling | ❌ 0% (training) |
| AND | Color masking | ✅ 100% |
| OR | Pixel combining | ✅ 100% |
| LSL | Address calc | ❌ 5% (training) |
| LSR | Fixed-point math | ❌ 10% (training) |

## Files to Create
1. `doom/neural_renderer.py` - DOOM-style raycaster using Neural CPU
2. `doom/wad_parser.py` - Parse DOOM WAD files
3. `doom/doom_demo.py` - Interactive demo
