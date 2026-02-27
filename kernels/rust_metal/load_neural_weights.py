#!/usr/bin/env python3
"""
Load trained PyTorch neural models and extract weights for GPU loading.

This script loads the trained neural network models and extracts their
weights into flat arrays that can be passed to the Rust Metal backend.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to find models
sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_state_dict_weights(state_dict, prefix_filter=""):
    """
    Extract all weights from a PyTorch state dict as a single flat array.

    Args:
        state_dict: PyTorch state_dict (OrderedDict)
        prefix_filter: Optional prefix to filter parameters

    Returns:
        flat_weights: numpy array of all weights
        shapes: dict mapping parameter names to shapes
    """
    weights_list = []
    shapes = {}

    for name, param in state_dict.items():
        # Skip if prefix filter doesn't match
        if prefix_filter and not name.startswith(prefix_filter):
            continue

        # Convert to numpy and flatten
        weight_array = param.detach().cpu().numpy().flatten()
        weights_list.append(weight_array)
        shapes[name] = param.shape

    # Concatenate all weights
    flat_weights = np.concatenate(weights_list).astype(np.float32)

    print(f"  Extracted {len(shapes)} parameters, {flat_weights.size} total weights")
    return flat_weights, shapes


def load_loop_detector_v2():
    """Load Loop Detector V2 model (LSTM+Attention for loop detection)."""
    print("\n" + "="*70)
    print("Loading Loop Detector V2...")
    print("="*70)

    model_path = Path(__file__).parent.parent / "loop_detector_v2.pt"

    if not model_path.exists():
        print(f"  ⚠️  Model not found: {model_path}")
        return None, None

    state_dict = torch.load(model_path, map_location='cpu')

    # Extract weights (excluding config if present)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    weights, shapes = extract_state_dict_weights(state_dict)

    print(f"  ✅ Loop Detector V2 loaded: {weights.size} params")
    return weights, shapes


def load_memory_oracle():
    """Load Memory Oracle model (LSTM-based memory access predictor)."""
    print("\n" + "="*70)
    print("Loading Memory Oracle...")
    print("="*70)

    model_path = Path(__file__).parent.parent / "memory_oracle_lstm.pt"

    if not model_path.exists():
        print(f"  ⚠️  Model not found: {model_path}")
        return None, None

    data = torch.load(model_path, map_location='cpu')

    # Handle different formats
    if isinstance(data, dict) and 'model_state_dict' in data:
        state_dict = data['model_state_dict']
        print(f"  Found model config: {data.get('config', 'N/A')}")
        print(f"  Training metrics: {data.get('metrics', 'N/A')}")
    else:
        state_dict = data

    weights, shapes = extract_state_dict_weights(state_dict)

    print(f"  ✅ Memory Oracle loaded: {weights.size} params")
    return weights, shapes


def load_symbol_resolver():
    """Load Symbol Resolver model (Transformer-based symbol resolution)."""
    print("\n" + "="*70)
    print("Loading Symbol Resolver...")
    print("="*70)

    model_path = Path(__file__).parent.parent / "symbol_resolver.pt"

    if not model_path.exists():
        print(f"  ⚠️  Model not found: {model_path}")
        return None, None

    state_dict = torch.load(model_path, map_location='cpu')

    weights, shapes = extract_state_dict_weights(state_dict)

    print(f"  ✅ Symbol Resolver loaded: {weights.size} params")
    return weights, shapes


def create_dispatch_weights():
    """
    Create default dispatch neural network weights (8→8→7).

    This is a simple network for kernel prediction. In production,
    this would be trained on execution traces.
    """
    print("\n" + "="*70)
    print("Creating Dispatch Network Weights (8→8→7)...")
    print("="*70)

    # Network architecture:
    # - Input: 8 features (opcode, instruction bytes, PC hash, etc.)
    # - Hidden: 8 neurons with tanh activation
    # - Output: 7 kernel types (arithmetic, logical, load/store, branch, etc.)

    np.random.seed(42)

    # Input layer: 8 inputs × 8 hidden = 64 weights + 8 biases = 72
    input_weights = np.random.randn(8, 8).astype(np.float32) * 0.1
    input_bias = np.zeros(8, dtype=np.float32)

    # Output layer: 8 hidden × 7 outputs = 56 weights + 7 biases = 63
    output_weights = np.random.randn(8, 7).astype(np.float32) * 0.1
    output_bias = np.zeros(7, dtype=np.float32)

    # Flatten: [W_input, b_input, W_output, b_output]
    flat_weights = np.concatenate([
        input_weights.flatten(),
        input_bias.flatten(),
        output_weights.flatten(),
        output_bias.flatten()
    ])

    print(f"  ✅ Dispatch network: {flat_weights.size} params")
    print(f"     Architecture: 8 inputs → 8 hidden → 7 outputs")
    return flat_weights


def main():
    """Load all models and prepare weights for GPU upload."""
    print("\n" + "="*70)
    print("  🧠 NEURAL MODEL WEIGHT LOADER")
    print("="*70)
    print("\nLoading trained PyTorch models for GPU acceleration...")

    # Load all models
    loop_weights, loop_shapes = load_loop_detector_v2()
    memory_weights, memory_shapes = load_memory_oracle()
    symbol_weights, symbol_shapes = load_symbol_resolver()
    dispatch_weights = create_dispatch_weights()

    # Summary
    print("\n" + "="*70)
    print("  📊 SUMMARY")
    print("="*70)

    total_params = 0
    if loop_weights is not None:
        total_params += loop_weights.size
        print(f"  Loop Detector V2:      {loop_weights.size:>12,} params")
    if memory_weights is not None:
        total_params += memory_weights.size
        print(f"  Memory Oracle:         {memory_weights.size:>12,} params")
    if symbol_weights is not None:
        total_params += symbol_weights.size
        print(f"  Symbol Resolver:       {symbol_weights.size:>12,} params")
    print(f"  Dispatch Network:      {dispatch_weights.size:>12,} params")
    print(f"  " + "-"*60)
    print(f"  TOTAL:                 {total_params + dispatch_weights.size:>12,} params")

    # Save weights to disk for loading by Rust
    output_dir = Path(__file__).parent.parent / "rust_metal" / "weights"
    output_dir.mkdir(exist_ok=True)

    if loop_weights is not None:
        np.save(output_dir / "loop_detector_weights.npy", loop_weights)
        print(f"\n  ✅ Saved: loop_detector_weights.npy")

    if memory_weights is not None:
        np.save(output_dir / "memory_oracle_weights.npy", memory_weights)
        print(f"  ✅ Saved: memory_oracle_weights.npy")

    if symbol_weights is not None:
        np.save(output_dir / "symbol_resolver_weights.npy", symbol_weights)
        print(f"  ✅ Saved: symbol_resolver_weights.npy")

    np.save(output_dir / "dispatch_weights.npy", dispatch_weights)
    print(f"  ✅ Saved: dispatch_weights.npy")

    print("\n" + "="*70)
    print("  ✅ All weights loaded and saved successfully!")
    print("="*70)
    print("\nNext step: Load these weights into NeuralMetalCPU via Python binding")
    print("="*70 + "\n")

    return {
        'loop_weights': loop_weights,
        'memory_weights': memory_weights,
        'symbol_weights': symbol_weights,
        'dispatch_weights': dispatch_weights,
    }


if __name__ == "__main__":
    weights = main()
