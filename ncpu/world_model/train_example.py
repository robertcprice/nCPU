"""
Simple end-to-end training example for the JEPA Machine World Model.

Usage:
    python -m ncpu.world_model.train_example
"""

from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader

from ncpu.world_model.je_world_model import JEWorldModel, JEWMConfig


def main():
    print("=== JEPA Machine World Model - End-to-End Training Example ===\n")

    # 1. Generate a small dataset on the fly (in real use you'd use generate_machine_traces)
    print("1. Generating synthetic traces (using current generator)...")
    from ncpu.world_model.generate_machine_traces import generate_single_trace

    all_transitions = []
    for _ in range(800):
        all_transitions.extend(generate_single_trace(max_steps=5))

    states_before = torch.stack([t[0] for t in all_transitions])
    actions = torch.stack([t[1] for t in all_transitions])
    states_after = torch.stack([t[2] for t in all_transitions])

    print(f"   {len(all_transitions)} transitions ready")

    # 2. Create model matching the feature size
    config = JEWMConfig(state_dim=22, action_dim=8, hidden_dim=64, num_predictor_layers=2)
    model = JEWorldModel(config)

    dataset = TensorDataset(states_before, actions, states_after)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 3. Train
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    print("\n2. Training for a few epochs...")
    for epoch in range(1, 6):
        total_loss = 0.0
        for s, a, s_next in loader:
            optimizer.zero_grad()
            out = model.forward_for_training(s, a, s_next)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"   Epoch {epoch}: loss = {total_loss / len(loader):.6f}")

    # 4. Save
    out_path = Path("jewm_example.pt")
    torch.save({"model_state": model.state_dict(), "config": config.to_dict()}, out_path)
    print(f"\n3. Saved example checkpoint to {out_path}")

    print("\nThis demonstrates the full loop: real machine traces → JEPA world model training.")
    print("Next: integrate the trained model into ExecutableThoughtHead for cheap speculation.")


if __name__ == "__main__":
    main()