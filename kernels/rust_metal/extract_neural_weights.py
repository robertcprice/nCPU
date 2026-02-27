#!/usr/bin/env python3
"""
Extract trained neural model weights and flatten for GPU loading.

This extracts the complex PyTorch state dicts and flattens them
into simple weight arrays that the GPU can load.
"""

import torch
import numpy as np
from pathlib import Path

def extract_loop_detector_v2_weights(model_path: str, output_path: str):
    """
    Extract Loop Detector V2 weights (1.08M params).

    Complex LSTM + Attention model with multiple components.
    """
    print("Loading Loop Detector V2 model...")
    model = torch.load(model_path, map_location='cpu')

    weights = []
    names = []

    # Flatten in predictable order for GPU shader
    # Order: inst_embed → reg_field_extract → seq_encoder → reg_embed → cross_attn → heads

    # Instruction embedding layers
    for key in ['inst_embed.0.weight', 'inst_embed.0.bias',
                  'inst_embed.1.weight', 'inst_embed.1.bias',
                  'inst_embed.3.weight', 'inst_embed.3.bias']:
        if key in model:
            w = model[key].detach().cpu().numpy().flatten()
            weights.append(w)
            names.append((key, w.shape[0]))

    # Register field extraction
    for key in ['reg_field_extract.weight', 'reg_field_extract.bias']:
        if key in model:
            w = model[key].detach().cpu().numpy().flatten()
            weights.append(w)
            names.append((key, w.shape[0]))

    # Sequential encoder (LSTM) - layer 0
    for key in ['seq_encoder.weight_ih_l0', 'seq_encoder.weight_hh_l0',
                  'seq_encoder.bias_ih_l0', 'seq_encoder.bias_hh_l0',
                  'seq_encoder.weight_ih_l0_reverse', 'seq_encoder.weight_hh_l0_reverse',
                  'seq_encoder.bias_ih_l0_reverse', 'seq_encoder.bias_hh_l0_reverse']:
        if key in model:
            w = model[key].detach().cpu().numpy().flatten()
            weights.append(w)
            names.append((key, w.shape[0]))

    # Sequential encoder (LSTM) - layer 1
    for key in ['seq_encoder.weight_ih_l1', 'seq_encoder.weight_hh_l1',
                  'seq_encoder.bias_ih_l1', 'seq_encoder.bias_hh_l1',
                  'seq_encoder.weight_ih_l1_reverse', 'seq_encoder.weight_hh_l1_reverse',
                  'seq_encoder.bias_ih_l1_reverse', 'seq_encoder.bias_hh_l1_reverse']:
        if key in model:
            w = model[key].detach().cpu().numpy().flatten()
            weights.append(w)
            names.append((key, w.shape[0]))

    # Register embedding
    for key in ['reg_embed.0.weight', 'reg_embed.0.bias',
                  'reg_embed.1.weight', 'reg_embed.1.bias',
                  'reg_embed.3.weight', 'reg_embed.3.bias']:
        if key in model:
            w = model[key].detach().cpu().numpy().flatten()
            weights.append(w)
            names.append((key, w.shape[0]))

    # Cross attention
    for key in ['cross_attn.in_proj_weight', 'cross_attn.in_proj_bias',
                  'cross_attn.out_proj.weight', 'cross_attn.out_proj.bias']:
        if key in model:
            w = model[key].detach().cpu().numpy().flatten()
            weights.append(w)
            names.append((key, w.shape[0]))

    # Type head
    for key in ['type_head.0.weight', 'type_head.0.bias',
                  'type_head.3.weight', 'type_head.3.bias']:
        if key in model:
            w = model[key].detach().cpu().numpy().flatten()
            weights.append(w)
            names.append((key, w.shape[0]))

    # Counter attention
    for key in ['counter_attn.weight', 'counter_attn.bias']:
        if key in model:
            w = model[key].detach().cpu().numpy().flatten()
            weights.append(w)
            names.append((key, w.shape[0]))

    # Iteration head
    for key in ['iter_head.0.weight', 'iter_head.0.bias',
                  'iter_head.2.weight', 'iter_head.2.bias']:
        if key in model:
            w = model[key].detach().cpu().numpy().flatten()
            weights.append(w)
            names.append((key, w.shape[0]))

    # Concatenate all weights
    flat_weights = np.concatenate(weights)

    print(f"✅ Extracted {len(names)} weight tensors")
    total_params = flat_weights.size
    print(f"✅ Total: {total_params:,} params ({total_params * 4 / 1024 / 1024:.2f} MB)")

    # Save
    np.save(output_path, flat_weights)
    print(f"✅ Saved to: {output_path}")

    # Save metadata
    metadata_path = output_path.replace('.npy', '_metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write(f"Loop Detector V2 - {total_params} params\n")
        f.write(f"Extracted from: {model_path}\n")
        f.write(f"\nWeight tensors ({len(names)}):\n")
        offset = 0
        for name, size in names:
            f.write(f"  [{offset:7d}:{offset+size:7d}] {name}\n")
            offset += size

    print(f"✅ Metadata saved to: {metadata_path}")

    return flat_weights

def extract_memory_oracle_weights(model_path: str, output_path: str):
    """
    Extract Memory Oracle LSTM weights (~271K params).
    """
    print("\nLoading Memory Oracle LSTM model...")
    model = torch.load(model_path, map_location='cpu')

    if 'model_state_dict' not in model:
        print("❌ Unexpected model format")
        return None

    state = model['model_state_dict']
    weights = []
    names = []

    # Flatten in order: feature_encoder → lstm → heads
    # Feature encoder
    for key in ['feature_encoder.0.weight', 'feature_encoder.0.bias',
                  'feature_encoder.2.weight', 'feature_encoder.2.bias']:
        if key in state:
            w = state[key].detach().cpu().numpy().flatten()
            weights.append(w)
            names.append((key, w.shape[0]))

    # LSTM layer 0
    for key in ['lstm.weight_ih_l0', 'lstm.weight_hh_l0',
                  'lstm.bias_ih_l0', 'lstm.bias_hh_l0',
                  'lstm.weight_ih_l1', 'lstm.weight_hh_l1',
                  'lstm.bias_ih_l1', 'lstm.bias_hh_l1']:
        if key in state:
            w = state[key].detach().cpu().numpy().flatten()
            weights.append(w)
            names.append((key, w.shape[0]))

    # Heads
    for key in ['delta_head.0.weight', 'delta_head.0.bias',
                  'delta_head.3.weight', 'delta_head.3.bias',
                  'confidence_head.0.weight', 'confidence_head.0.bias',
                  'confidence_head.2.weight', 'confidence_head.2.bias',
                  'pattern_head.0.weight', 'pattern_head.0.bias',
                  'pattern_head.2.weight', 'pattern_head.2.bias']:
        if key in state:
            w = state[key].detach().cpu().numpy().flatten()
            weights.append(w)
            names.append((key, w.shape[0]))

    # Concatenate
    flat_weights = np.concatenate(weights)

    print(f"✅ Extracted {len(names)} weight tensors")
    total_params = flat_weights.size
    print(f"✅ Total: {total_params:,} params ({total_params * 4 / 1024 / 1024:.2f} MB)")

    # Save
    np.save(output_path, flat_weights)
    print(f"✅ Saved to: {output_path}")

    # Metadata
    metadata_path = output_path.replace('.npy', '_metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write(f"Memory Oracle LSTM - {total_params} params\n")
        f.write(f"Extracted from: {model_path}\n")
        f.write(f"\nWeight tensors ({len(names)}):\n")
        offset = 0
        for name, size in names:
            f.write(f"  [{offset:7d}:{offset+size:7d}] {name}\n")
            offset += size

    print(f"✅ Metadata saved to: {metadata_path}")

    return flat_weights

def main():
    base_dir = Path("/Users/bobbyprice/projects/KVRM/kvrm-cpu")
    output_dir = Path(__file__).parent / "weights"
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("  🔧 EXTRACTING NEURAL MODEL WEIGHTS FOR GPU")
    print("=" * 80)
    print()

    # Extract Loop Detector V2
    loop_detector_path = base_dir / "loop_detector_v2.pt"
    if loop_detector_path.exists():
        extract_loop_detector_v2_weights(
            str(loop_detector_path),
            str(output_dir / "loop_detector_v2_weights.npy")
        )
    else:
        print(f"⚠️  Loop Detector V2 not found: {loop_detector_path}")

    # Extract Memory Oracle
    memory_oracle_path = base_dir / "memory_oracle_lstm.pt"
    if memory_oracle_path.exists():
        extract_memory_oracle_weights(
            str(memory_oracle_path),
            str(output_dir / "memory_oracle_lstm_weights.npy")
        )
    else:
        print(f"⚠️  Memory Oracle not found: {memory_oracle_path}")

    print()
    print("=" * 80)
    print("  ✅ EXTRACTION COMPLETE")
    print("=" * 80)
    print()
    print("Generated files:")
    print(f"  - weights/loop_detector_v2_weights.npy")
    print(f"  - weights/loop_detector_v2_weights_metadata.txt")
    print(f"  - weights/memory_oracle_lstm_weights.npy")
    print(f"  - weights/memory_oracle_lstm_weights_metadata.txt")

if __name__ == "__main__":
    main()
