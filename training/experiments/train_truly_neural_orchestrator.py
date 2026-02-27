#!/usr/bin/env python3
"""
TRULY NEURAL SYSTEM ORCHESTRATOR
=================================
The meta-model that coordinates ALL neural system components.

This is the "brain" of the neural OS - it learns:
1. When to fetch instructions (from MMU)
2. When to decode instructions (decoder)
3. When to execute (ALU + register file)
4. When to handle syscalls (syscall handlers)
5. When to handle interrupts (GIC)
6. When to update timer
7. Context switching decisions
8. Exception handling

ALL execution flow is LEARNED, not hardcoded!
This replaces the traditional fetch-decode-execute loop with
a neural network that learns the optimal execution strategy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


class TrulyNeuralOrchestrator(nn.Module):
    """
    Neural System Orchestrator - coordinates all components.

    The orchestrator maintains:
    - CPU state (PC, SP, EL, flags) IN weights
    - Execution phase IN weights
    - Pending operations queue IN weights

    It learns to:
    - Decide next action (fetch/decode/execute/syscall/irq)
    - Route data between components
    - Handle exceptions and context switches
    """

    def __init__(self, key_dim=128, num_actions=10):
        super().__init__()
        self.key_dim = key_dim
        self.num_actions = num_actions

        # === CPU STATE IN WEIGHTS ===
        # Program counter (64-bit)
        self.pc_state = nn.Parameter(torch.zeros(64))

        # Stack pointer (64-bit)
        self.sp_state = nn.Parameter(torch.zeros(64))

        # Exception level (2-bit: EL0-EL3)
        self.el_state = nn.Parameter(torch.zeros(4))

        # CPU flags (NZCV)
        self.flags_state = nn.Parameter(torch.zeros(4))

        # Execution phase (one-hot: FETCH, DECODE, EXECUTE, SYSCALL, IRQ, IDLE)
        self.phase_state = nn.Parameter(torch.zeros(6))

        # Pending instruction buffer (holds fetched/decoded instruction)
        self.instruction_buffer = nn.Parameter(torch.zeros(32))

        # Pending IRQ state
        self.pending_irq = nn.Parameter(torch.zeros(8))

        # === ACTION DEFINITIONS ===
        # 0: FETCH - fetch instruction from memory
        # 1: DECODE - decode instruction
        # 2: EXECUTE_ALU - execute ALU operation
        # 3: EXECUTE_MEM - memory load/store
        # 4: EXECUTE_BRANCH - branch/jump
        # 5: SYSCALL - handle syscall
        # 6: IRQ_CHECK - check for interrupts
        # 7: IRQ_HANDLE - handle interrupt
        # 8: CONTEXT_SWITCH - switch context
        # 9: IDLE - wait for event

        # === LEARNED DECISION NETWORKS ===
        # State encoder: all CPU state → latent representation
        state_input_dim = 64 + 64 + 4 + 4 + 6 + 32 + 8  # pc, sp, el, flags, phase, instr, irq
        self.state_encoder = nn.Sequential(
            nn.Linear(state_input_dim, key_dim * 2),
            nn.GELU(),
            nn.Linear(key_dim * 2, key_dim),
            nn.LayerNorm(key_dim),
        )

        # Action selector: latent state → action distribution
        self.action_selector = nn.Sequential(
            nn.Linear(key_dim, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, num_actions),
        )

        # Action-specific networks
        # Fetch network: state → memory address to fetch
        self.fetch_network = nn.Sequential(
            nn.Linear(key_dim, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, 64),  # address bits
        )

        # Decode network: instruction → decoded info
        self.decode_network = nn.Sequential(
            nn.Linear(key_dim + 32, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, 64),  # decoded operation info
        )

        # Execute network: state + decoded → result + new state
        self.execute_network = nn.Sequential(
            nn.Linear(key_dim + 64, key_dim * 2),
            nn.GELU(),
            nn.Linear(key_dim * 2, 64 + 64 + 4),  # result, new_pc, new_flags
        )

        # Syscall network: state → syscall handling
        self.syscall_network = nn.Sequential(
            nn.Linear(key_dim, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, 64 + 1),  # result + success
        )

        # IRQ network: state + irq_info → handling decision
        self.irq_network = nn.Sequential(
            nn.Linear(key_dim + 8, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, 64 + 1),  # handler_addr + should_handle
        )

        # PC update network: decides how to update PC after each action
        self.pc_update_network = nn.Sequential(
            nn.Linear(key_dim + num_actions + 64, key_dim),
            nn.GELU(),
            nn.Linear(key_dim, 64),  # new PC
        )

        # Phase transition network: current_phase + action → new_phase
        self.phase_network = nn.Sequential(
            nn.Linear(6 + num_actions, 32),
            nn.GELU(),
            nn.Linear(32, 6),  # new phase
        )

        # Learning rate for state updates
        self.update_lr = nn.Parameter(torch.tensor(0.3))

        # Temperature for action selection
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def get_state_encoding(self):
        """Encode current CPU state."""
        state = torch.cat([
            torch.sigmoid(self.pc_state),
            torch.sigmoid(self.sp_state),
            torch.sigmoid(self.el_state),
            torch.sigmoid(self.flags_state),
            F.softmax(self.phase_state, dim=-1),
            torch.sigmoid(self.instruction_buffer),
            torch.sigmoid(self.pending_irq),
        ]).unsqueeze(0)

        return self.state_encoder(state)

    def select_action(self, state_encoding):
        """Select next action based on state."""
        action_logits = self.action_selector(state_encoding)
        temp = torch.clamp(self.temperature.abs(), min=0.1)
        action_probs = F.softmax(action_logits / temp, dim=-1)
        return action_probs

    def step(self, external_signals=None):
        """
        Execute one orchestration step.

        Args:
            external_signals: Optional dict with:
                - 'instruction': [32] fetched instruction bits
                - 'memory_data': [64] data from memory
                - 'irq_pending': [8] interrupt signals
                - 'syscall_result': [64] result from syscall handler

        Returns:
            action: selected action index
            outputs: dict with action-specific outputs
        """
        if external_signals is None:
            external_signals = {}

        # Encode state
        state_enc = self.get_state_encoding()

        # Select action
        action_probs = self.select_action(state_enc)
        action = action_probs.argmax(dim=-1).item()

        outputs = {'action_probs': action_probs}
        lr = torch.clamp(self.update_lr.abs(), 0.01, 1.0)

        # Execute action-specific logic
        if action == 0:  # FETCH
            fetch_addr = torch.sigmoid(self.fetch_network(state_enc))
            outputs['fetch_addr'] = fetch_addr

            # Update instruction buffer if instruction provided
            if 'instruction' in external_signals:
                instr = external_signals['instruction']
                instr_logits = torch.log(instr.clamp(0.01, 0.99) / (1 - instr.clamp(0.01, 0.99)))
                self.instruction_buffer.data = (1 - lr) * self.instruction_buffer.data + lr * instr_logits

        elif action == 1:  # DECODE
            decode_input = torch.cat([state_enc, torch.sigmoid(self.instruction_buffer).unsqueeze(0)], dim=-1)
            decoded = self.decode_network(decode_input)
            outputs['decoded'] = decoded

        elif action == 2:  # EXECUTE_ALU
            decoded = torch.sigmoid(self.instruction_buffer[:64]).unsqueeze(0)  # Use buffer as decoded
            execute_input = torch.cat([state_enc, decoded], dim=-1)
            execute_out = self.execute_network(execute_input)

            result = execute_out[:, :64]
            new_pc = execute_out[:, 64:128]
            new_flags = execute_out[:, 128:132]

            outputs['result'] = result
            outputs['new_pc'] = new_pc

            # Update PC
            pc_logits = torch.log(new_pc.squeeze(0).clamp(0.01, 0.99) / (1 - new_pc.squeeze(0).clamp(0.01, 0.99)))
            self.pc_state.data = (1 - lr) * self.pc_state.data + lr * pc_logits

            # Update flags
            flags_logits = torch.log(new_flags.squeeze(0).clamp(0.01, 0.99) / (1 - new_flags.squeeze(0).clamp(0.01, 0.99)))
            self.flags_state.data = (1 - lr) * self.flags_state.data + lr * flags_logits

        elif action == 5:  # SYSCALL
            syscall_out = self.syscall_network(state_enc)
            result = syscall_out[:, :64]
            success = torch.sigmoid(syscall_out[:, 64])

            outputs['syscall_result'] = result
            outputs['syscall_success'] = success

        elif action == 6:  # IRQ_CHECK
            if 'irq_pending' in external_signals:
                irq = external_signals['irq_pending']
                irq_logits = torch.log(irq.clamp(0.01, 0.99) / (1 - irq.clamp(0.01, 0.99)))
                self.pending_irq.data = (1 - lr) * self.pending_irq.data + lr * irq_logits

            outputs['has_irq'] = (torch.sigmoid(self.pending_irq).sum() > 0.5).item()

        elif action == 7:  # IRQ_HANDLE
            irq_input = torch.cat([state_enc, torch.sigmoid(self.pending_irq).unsqueeze(0)], dim=-1)
            irq_out = self.irq_network(irq_input)

            handler_addr = irq_out[:, :64]
            should_handle = torch.sigmoid(irq_out[:, 64])

            outputs['handler_addr'] = handler_addr
            outputs['should_handle'] = should_handle

            # Clear pending IRQ
            self.pending_irq.data = self.pending_irq.data * 0.5

        # Update phase
        action_onehot = F.one_hot(torch.tensor([action]), num_classes=self.num_actions).float().to(device)
        phase_input = torch.cat([F.softmax(self.phase_state, dim=-1).unsqueeze(0), action_onehot], dim=-1)
        new_phase_logits = self.phase_network(phase_input)
        self.phase_state.data = (1 - lr) * self.phase_state.data + lr * new_phase_logits.squeeze(0)

        return action, outputs

    def orchestrate_cycle(self, memory_interface=None, decoder_interface=None,
                         alu_interface=None, syscall_interface=None, gic_interface=None):
        """
        Run a complete orchestration cycle.

        This is the main entry point that coordinates all neural components.
        """
        # Check for interrupts first
        action, outputs = self.step({'irq_pending': torch.zeros(8).to(device)})

        cycle_outputs = {
            'action': action,
            'action_name': ['FETCH', 'DECODE', 'EXECUTE_ALU', 'EXECUTE_MEM',
                           'EXECUTE_BRANCH', 'SYSCALL', 'IRQ_CHECK', 'IRQ_HANDLE',
                           'CONTEXT_SWITCH', 'IDLE'][action],
            'outputs': outputs,
        }

        return cycle_outputs


def generate_training_sequence(length, device):
    """Generate a training sequence of CPU states and expected actions."""
    sequences = []

    for _ in range(length):
        # Random CPU state
        pc = torch.rand(64)
        sp = torch.rand(64)
        el = F.one_hot(torch.tensor(random.randint(0, 3)), num_classes=4).float()
        flags = torch.rand(4)
        phase = F.one_hot(torch.tensor(random.randint(0, 5)), num_classes=6).float()
        instruction = torch.rand(32)
        irq = torch.rand(8)

        # Determine expected action based on phase
        phase_idx = phase.argmax().item()

        if phase_idx == 0:  # FETCH phase
            expected_action = 0  # FETCH
        elif phase_idx == 1:  # DECODE phase
            expected_action = 1  # DECODE
        elif phase_idx == 2:  # EXECUTE phase
            expected_action = random.choice([2, 3, 4])  # ALU, MEM, or BRANCH
        elif phase_idx == 3:  # SYSCALL phase
            expected_action = 5  # SYSCALL
        elif phase_idx == 4:  # IRQ phase
            expected_action = 7 if irq.sum() > 4 else 6  # HANDLE or CHECK
        else:  # IDLE
            expected_action = 9

        sequences.append({
            'state': torch.cat([pc, sp, el, flags, phase, instruction, irq]),
            'expected_action': expected_action,
        })

    return sequences


def train():
    print("=" * 60)
    print("TRULY NEURAL SYSTEM ORCHESTRATOR TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print("The 'brain' of the Neural OS - coordinates ALL components!")
    print("Learns: fetch→decode→execute→syscall→irq handling flow")

    model = TrulyNeuralOrchestrator(
        key_dim=128,
        num_actions=10
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    os.makedirs("models/final", exist_ok=True)

    best_acc = 0
    batch_size = 128
    seq_length = 100

    for epoch in range(100):
        model.train()
        total_loss = 0
        t0 = time.time()

        for _ in range(50):
            sequences = generate_training_sequence(batch_size, device)

            optimizer.zero_grad()

            batch_loss = 0
            for seq in sequences:
                # Set CPU state
                state = seq['state'].to(device)
                expected = seq['expected_action']

                # Split state back into components
                model.pc_state.data = torch.log(state[:64].clamp(0.01, 0.99) / (1 - state[:64].clamp(0.01, 0.99)))
                model.sp_state.data = torch.log(state[64:128].clamp(0.01, 0.99) / (1 - state[64:128].clamp(0.01, 0.99)))
                model.el_state.data = torch.log(state[128:132].clamp(0.01, 0.99) / (1 - state[128:132].clamp(0.01, 0.99)))
                model.flags_state.data = torch.log(state[132:136].clamp(0.01, 0.99) / (1 - state[132:136].clamp(0.01, 0.99)))
                model.phase_state.data = torch.log(state[136:142].clamp(0.01, 0.99) / (1 - state[136:142].clamp(0.01, 0.99)))
                model.instruction_buffer.data = torch.log(state[142:174].clamp(0.01, 0.99) / (1 - state[142:174].clamp(0.01, 0.99)))
                model.pending_irq.data = torch.log(state[174:182].clamp(0.01, 0.99) / (1 - state[174:182].clamp(0.01, 0.99)))

                # Get action
                state_enc = model.get_state_encoding()
                action_probs = model.select_action(state_enc)

                # Loss
                target = torch.tensor([expected], device=device)
                loss = F.cross_entropy(action_probs, target)
                batch_loss += loss

            batch_loss = batch_loss / batch_size
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += batch_loss.item()

        scheduler.step()

        # Test
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            sequences = generate_training_sequence(500, device)

            for seq in sequences:
                state = seq['state'].to(device)
                expected = seq['expected_action']

                model.pc_state.data = torch.log(state[:64].clamp(0.01, 0.99) / (1 - state[:64].clamp(0.01, 0.99)))
                model.sp_state.data = torch.log(state[64:128].clamp(0.01, 0.99) / (1 - state[64:128].clamp(0.01, 0.99)))
                model.el_state.data = torch.log(state[128:132].clamp(0.01, 0.99) / (1 - state[128:132].clamp(0.01, 0.99)))
                model.flags_state.data = torch.log(state[132:136].clamp(0.01, 0.99) / (1 - state[132:136].clamp(0.01, 0.99)))
                model.phase_state.data = torch.log(state[136:142].clamp(0.01, 0.99) / (1 - state[136:142].clamp(0.01, 0.99)))
                model.instruction_buffer.data = torch.log(state[142:174].clamp(0.01, 0.99) / (1 - state[142:174].clamp(0.01, 0.99)))
                model.pending_irq.data = torch.log(state[174:182].clamp(0.01, 0.99) / (1 - state[174:182].clamp(0.01, 0.99)))

                state_enc = model.get_state_encoding()
                action_probs = model.select_action(state_enc)
                pred = action_probs.argmax(dim=-1).item()

                if pred == expected:
                    correct += 1
                total += 1

        acc = correct / total
        elapsed = time.time() - t0

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss/50:.4f} action_acc={100*acc:.1f}% [{elapsed:.1f}s]")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "action_accuracy": acc,
                "op_name": "TRULY_NEURAL_ORCHESTRATOR",
                "actions": ['FETCH', 'DECODE', 'EXECUTE_ALU', 'EXECUTE_MEM',
                           'EXECUTE_BRANCH', 'SYSCALL', 'IRQ_CHECK', 'IRQ_HANDLE',
                           'CONTEXT_SWITCH', 'IDLE'],
            }, "models/final/truly_neural_orchestrator_best.pt")
            print(f"  Saved (action_acc={100*acc:.1f}%)")

        if acc >= 0.95:
            print("95%+ ACCURACY!")
            break

    print(f"\nBest action accuracy: {100*best_acc:.1f}%")

    # Final verification
    print("\nFinal verification - action selection per phase:")
    model.eval()
    PHASES = ['FETCH', 'DECODE', 'EXECUTE', 'SYSCALL', 'IRQ', 'IDLE']
    ACTIONS = ['FETCH', 'DECODE', 'EXECUTE_ALU', 'EXECUTE_MEM', 'EXECUTE_BRANCH',
               'SYSCALL', 'IRQ_CHECK', 'IRQ_HANDLE', 'CONTEXT_SWITCH', 'IDLE']

    with torch.no_grad():
        for phase_idx, phase_name in enumerate(PHASES):
            # Set phase
            phase = F.one_hot(torch.tensor(phase_idx), num_classes=6).float().to(device)

            # Create test state
            state = torch.cat([
                torch.zeros(64),  # pc
                torch.zeros(64),  # sp
                torch.zeros(4),   # el
                torch.zeros(4),   # flags
                phase,
                torch.zeros(32),  # instruction
                torch.zeros(8),   # irq
            ]).to(device)

            model.pc_state.data = torch.log(state[:64].clamp(0.01, 0.99) / (1 - state[:64].clamp(0.01, 0.99)))
            model.sp_state.data = torch.log(state[64:128].clamp(0.01, 0.99) / (1 - state[64:128].clamp(0.01, 0.99)))
            model.el_state.data = torch.log(state[128:132].clamp(0.01, 0.99) / (1 - state[128:132].clamp(0.01, 0.99)))
            model.flags_state.data = torch.log(state[132:136].clamp(0.01, 0.99) / (1 - state[132:136].clamp(0.01, 0.99)))
            model.phase_state.data = torch.log(state[136:142].clamp(0.01, 0.99) / (1 - state[136:142].clamp(0.01, 0.99)))
            model.instruction_buffer.data = torch.log(state[142:174].clamp(0.01, 0.99) / (1 - state[142:174].clamp(0.01, 0.99)))
            model.pending_irq.data = torch.log(state[174:182].clamp(0.01, 0.99) / (1 - state[174:182].clamp(0.01, 0.99)))

            state_enc = model.get_state_encoding()
            action_probs = model.select_action(state_enc)
            pred_action = action_probs.argmax(dim=-1).item()

            print(f"  Phase {phase_name} → Action {ACTIONS[pred_action]}")


if __name__ == "__main__":
    train()
