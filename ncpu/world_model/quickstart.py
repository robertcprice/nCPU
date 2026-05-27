"""
Quickstart for the JEPA Machine World Model work.

Run this after `pip install -e ".[demo]"` (or with torch available):

    python -m ncpu.world_model.quickstart
"""

from pathlib import Path
import torch

from ncpu.world_model.je_world_model import create_small_jewm
from ncpu.world_model.generate_machine_traces import generate_single_trace
from ncpu.differentiable.execution import DifferentiableEngine


def main():
    print("=== JEPA Machine World Model Quickstart ===\n")

    print("1. Creating small JEPA world model (matching 22-dim prototype traces)...")
    from ncpu.world_model.je_world_model import JEWMConfig, JEWorldModel
    cfg = JEWMConfig(state_dim=22, action_dim=8, hidden_dim=64, num_predictor_layers=2)
    model = JEWorldModel(cfg)
    print(f"   Model created with state_dim={model.config.state_dim}")

    print("\n2. Generating a few synthetic machine traces using DifferentiableEngine...")
    traces = []
    for _ in range(3):
        t = generate_single_trace(max_steps=4, num_registers=8)
        traces.extend(t)
    print(f"   Generated {len(traces)} (state, action, next) transitions")

    if traces:
        print("\n3. Running a forward pass through the world model...")
        s, a, _ = traces[0]
        with torch.no_grad():
            pred = model.predict_next_latent(
                model.encode_state(s.unsqueeze(0)),
                model.encode_action(a.unsqueeze(0))
            )
        print(f"   Predicted next latent shape: {pred.shape}")

    print("\n4. Next steps:")
    print("   - Generate real dataset: python -m ncpu.world_model.generate_machine_traces --num-traces 2000")
    print("   - Train model:          python -m ncpu.world_model.train_je_world_model --data machine_traces.pt")
    print("   - See design + integration: docs/architecture/JEPA_MACHINE_WORLD_MODEL.md")
    print("\nThe speculate_with_world_model hook on ExecutableThoughtHead now actually uses the JEWorldModel for cheap rollouts (see integration_example.py).")
    print("This is the foundation for fast latent simulation + robustness on top of the exact neural GPU computer.")
    print("\n(Deep continuous grind in progress — cleaner repo, stronger hero, more real JEPA every pass.)")


if __name__ == "__main__":
    main()