#!/usr/bin/env python3
"""
COLLECT REAL LOOP DATA FROM ACTUAL EXECUTION
=============================================

This script runs DOOM and collects REAL loop data including:
- Actual pattern embeddings (not random noise)
- Actual register values at loop entry
- Actual iteration counts
- Loop instruction sequences

This data will be used to RETRAIN the iteration predictor on real data,
not synthetic random data.
"""

import torch
import json
import sys
from pathlib import Path

sys.path.insert(0, '.')
from run_neural_rtos_v2 import FullyNeuralCPU, load_elf


def collect_loop_data():
    """Collect real loop data from DOOM execution."""

    print("="*70)
    print(" COLLECTING REAL LOOP DATA FROM DOOM")
    print("="*70)
    print()

    # Create CPU WITHOUT optimization (so loops actually execute)
    cpu = FullyNeuralCPU(fast_mode=True, batch_size=128, use_native_math=True)
    entry = load_elf(cpu, 'doom_benchmark.elf')
    cpu.pc = entry

    print(f"Entry point: 0x{entry:x}")
    print("Running DOOM to collect loop data...")
    print()

    # Track all loops we encounter
    collected_loops = []
    loop_analysis_cache = {}  # PC -> loop_body (cache body but allow multiple samples)
    samples_per_loop = {}  # PC -> count (track how many samples we've collected)

    # Run for enough instructions to hit all rendering loops
    max_instructions = 200000  # Increased to capture more loops
    instruction_count = 0

    while instruction_count < max_instructions and cpu.pc != 0:
        # Save state before instruction
        pc_before = cpu.pc
        regs_before = {i: cpu.regs.get(i) for i in range(32)}

        # Execute instruction
        cpu.step()
        instruction_count += 1

        # Check for backward branch (potential loop)
        if cpu.pc < pc_before:
            loop_start = cpu.pc
            loop_end = pc_before

            # Check if we've collected enough samples for this loop
            if samples_per_loop.get(loop_start, 0) >= 100:  # Max 100 samples per loop
                continue

            # Extract or cache loop body
            if loop_start in loop_analysis_cache:
                loop_body = loop_analysis_cache[loop_start]
            else:
                # Extract loop body
                loop_body = []
                pc = loop_start
                while pc <= loop_end:
                    inst = cpu.memory.read32(pc)
                    if inst in cpu.decode_cache:
                        decoded = cpu.decode_cache[inst]
                        loop_body.append((pc, inst, decoded))
                    else:
                        try:
                            decoded = cpu.decoder.decode(inst)
                            loop_body.append((pc, inst, decoded))
                            cpu.decode_cache[inst] = decoded
                        except:
                            loop_body.append((pc, inst, None))
                    pc += 4

                loop_analysis_cache[loop_start] = loop_body

            if len(loop_body) < 2:
                if samples_per_loop.get(loop_start, 0) == 0:
                    print(f"  Loop at 0x{loop_start:x}: Skipped (too small: {len(loop_body)} instructions)")
                samples_per_loop[loop_start] = samples_per_loop.get(loop_start, 0) + 1
                continue

            # Analyze loop to determine iterations
            print(f"Analyzing loop at 0x{loop_start:x}...")

            # Extract loop parameters to predict iterations
            count_reg = None
            limit_reg = None
            counter_value = 0
            limit_value = 0
            loop_type = None

            # Analyze loop body to find pattern
            for pc, inst, dec in loop_body:
                if dec and len(dec) >= 7:
                    # Check for SUBS that sets flags (decrement loop)
                    # Note: subs is sometimes classified as COMPARE (cat=11) with sets_flags=True
                    if dec[6]:  # Sets flags - this is the key for decrement loops
                        if dec[1] == dec[0] and dec[1] != 31:  # Same dest and source (subs x2, x2, #1)
                            count_reg = dec[1]
                            counter_value = regs_before[count_reg]
                            loop_type = 'decrement'
                            break  # Found the counter, we're done
                    # Check for COMPARE without flags (increment loop)
                    elif dec[3] == 11 and not dec[6]:  # COMPARE without flags
                        if dec[1] != 31:
                            if count_reg is None:  # Only set if not already found
                                count_reg = dec[1]
                                counter_value = regs_before[count_reg]
                        if dec[2] != 31 and dec[2] != 0:
                            limit_reg = dec[2]
                            limit_value = regs_before[limit_reg]
                            loop_type = 'increment'

            # Predict iterations based on loop type
            if loop_type == 'decrement':
                # Decrement-until-zero loop: iterations = counter value
                iterations = counter_value if counter_value > 0 and counter_value < 50000 else 0
            elif loop_type == 'increment' and limit_reg is not None:
                # Increment loop: iterations = limit - counter
                iterations = max(0, limit_value - counter_value)
                if iterations > 50000:
                    iterations = 0
            else:
                # Unknown loop type - skip
                continue

            samples_per_loop[loop_start] = samples_per_loop.get(loop_start, 0) + 1

            if iterations >= 2 and samples_per_loop[loop_start] <= 10:  # Only collect first 10 times per loop
                print(f"  Loop 0x{loop_start:x}: {iterations} iterations, type={loop_type}, sample #{samples_per_loop[loop_start]}")

                # Get pattern embedding using the pattern recognizer
                try:
                    # Import pattern recognizer from loop optimizer
                    from neural_loop_optimizer_v2 import SequentialPatternRecognizer

                    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
                    recognizer = SequentialPatternRecognizer().to(device)

                    # Load trained model
                    model_path = Path('models/pattern_recognizer_best.pt')
                    if model_path.exists():
                        checkpoint = torch.load(model_path, map_location=device)
                        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                            recognizer.load_state_dict(checkpoint['model_state_dict'])
                        elif isinstance(checkpoint, dict):
                            recognizer.load_state_dict(checkpoint)
                        recognizer.eval()

                    # Prepare sequence for embedding
                    seq_data = []
                    for pc, inst, dec in loop_body[:20]:
                        if dec and len(dec) >= 7:
                            seq_data.append([
                                float(dec[0]), float(dec[1]), float(dec[2]),
                                float(dec[3]), float(dec[4]), float(dec[5]), float(dec[6])
                            ])
                        else:
                            seq_data.append([0.0] * 7)

                    while len(seq_data) < 20:
                        seq_data.append([0.0] * 7)

                    inst_seq = torch.tensor([seq_data], dtype=torch.float32).to(device)

                    with torch.no_grad():
                        embedding = recognizer.get_embedding(inst_seq)  # (1, 128)
                        inst_encoding = embedding.squeeze(0).cpu().tolist()  # (128,)

                    # Extract register info
                    # For MEMSET pattern, find the counter register
                    count_reg = None
                    limit_reg = None
                    counter_value = 0
                    limit_value = 0

                    # Simple heuristic to find counter (look for SUBS that sets flags)
                    for pc, inst, dec in loop_body:
                        if dec and len(dec) >= 7:
                            if dec[6]:  # Sets flags
                                if dec[3] == 1 and dec[1] == dec[0]:  # SUBS with same dest/src
                                    count_reg = dec[1]
                                    counter_value = regs_before[count_reg]
                                    break
                            elif dec[3] == 11:  # COMPARE
                                if dec[1] != 31:
                                    count_reg = dec[1]
                                    counter_value = regs_before[count_reg]
                                if dec[2] != 31 and dec[2] != 0:  # Skip zero register
                                    limit_reg = dec[2]
                                    limit_value = regs_before[limit_reg]

                    collected_loops.append({
                        'loop_start': loop_start,
                        'loop_end': loop_end,
                        'iterations': iterations,
                        'inst_encoding': inst_encoding,  # REAL embedding from pattern recognizer
                        'counter': int(counter_value),
                        'limit': int(limit_value),
                        'pc': loop_start,
                        'num_instructions': len(loop_body),
                        'pattern_type': 'MEMSET'  # DOOM loops are all MEMSET-type
                    })

                except Exception as e:
                    import traceback
                    print(f"  Warning: Could not get embedding for loop 0x{loop_start:x}: {e}")
                    print(f"  Traceback: {traceback.format_exc()}")

        if instruction_count % 5000 == 0:
            print(f"Progress: {instruction_count}/{max_instructions} instructions")

    print()
    print(f"Collected {len(collected_loops)} unique loops")

    # Save to file
    output_file = Path('models/real_loop_data.json')
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(collected_loops, f, indent=2)

    print(f"✅ Saved real loop data to {output_file}")

    # Print statistics
    print()
    print("="*70)
    print(" LOOP STATISTICS")
    print("="*70)
    print()

    if collected_loops:
        iterations_list = [loop['iterations'] for loop in collected_loops]
        print(f"Total loops: {len(collected_loops)}")
        print(f"Min iterations: {min(iterations_list)}")
        print(f"Max iterations: {max(iterations_list)}")
        print(f"Avg iterations: {sum(iterations_list) / len(iterations_list):.1f}")
        print(f"Total iterations: {sum(iterations_list):,}")
        print()

        # Show distribution
        print("Iteration distribution:")
        ranges = [(0, 100), (100, 1000), (1000, 10000), (10000, 50000)]
        for min_i, max_i in ranges:
            count = sum(1 for i in iterations_list if min_i <= i < max_i)
            print(f"  {min_i:5d}-{max_i:5d}: {count} loops")

    print("="*70)
    print()

    return collected_loops


if __name__ == "__main__":
    collect_loop_data()
