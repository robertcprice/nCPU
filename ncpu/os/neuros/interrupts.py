"""Neural Generic Interrupt Controller (GIC).

GPU-native interrupt handling where priority ordering and dispatch routing
are performed by a neural network. Instead of fixed priority encoders,
the GIC learns optimal interrupt dispatch based on system state.

Architecture:
    - Interrupt request register (IRR): GPU tensor bitmap
    - In-service register (ISR): tracks active interrupt handlers
    - Neural priority encoder: MLP that scores pending interrupts
    - Dispatch table: maps IRQ numbers to handler functions

Key design: interrupts are tensors. Raising an interrupt sets a bit in
the IRR tensor. The neural priority encoder scores all pending interrupts
and selects the highest-priority one for dispatch.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Callable, Dict, List, Tuple
from pathlib import Path

from .device import default_device

logger = logging.getLogger(__name__)

# Standard IRQ assignments
IRQ_TIMER = 0
IRQ_KEYBOARD = 1
IRQ_DISK = 2
IRQ_NETWORK = 3
IRQ_IPC = 4
IRQ_PAGE_FAULT = 5
IRQ_SYSCALL = 6
IRQ_GPU = 7
NUM_IRQS = 32


class NeuralPriorityEncoder(nn.Module):
    """Neural network that learns interrupt priority ordering.

    Given the current interrupt state (which IRQs are pending, which are
    in service, system load), produces a priority score for each IRQ.

    Architecture:
        Input: [NUM_IRQS * 3] — pending bits + in_service bits + mask bits
        MLP: → hidden → scores
        Output: [NUM_IRQS] priority scores (higher = handle first)
    """

    def __init__(self, num_irqs: int = NUM_IRQS, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_irqs * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_irqs),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: [num_irqs * 3] → [num_irqs] priority scores"""
        return self.net(state)


InterruptHandler = Callable[[int], None]


class NeuralGIC:
    """Neural Generic Interrupt Controller.

    GPU-tensor interrupt controller with neural priority dispatch.
    All interrupt state lives on GPU:
        - irr: Interrupt Request Register (pending interrupts)
        - isr: In-Service Register (currently handling)
        - imr: Interrupt Mask Register (disabled interrupts)

    Dispatch loop:
        1. Check IRR for pending interrupts
        2. Neural encoder scores pending IRQs
        3. Highest-scoring unmasked, non-in-service IRQ is dispatched
        4. Handler runs, IRQ cleared from IRR and ISR
    """

    def __init__(self, num_irqs: int = NUM_IRQS,
                 device: Optional[torch.device] = None):
        self.device = device or default_device()
        self.num_irqs = num_irqs

        # Interrupt state (GPU tensors)
        self.irr = torch.zeros(num_irqs, dtype=torch.bool, device=self.device)
        self.isr = torch.zeros(num_irqs, dtype=torch.bool, device=self.device)
        self.imr = torch.zeros(num_irqs, dtype=torch.bool, device=self.device)  # True = masked

        # Fixed priority table (fallback)
        self.priority = torch.arange(num_irqs, dtype=torch.float32, device=self.device)
        # Lower IRQ number = higher priority by default

        # Neural priority encoder
        self.encoder = NeuralPriorityEncoder(num_irqs).to(self.device)
        self._trained = False

        # Handler dispatch table
        self._handlers: Dict[int, InterruptHandler] = {}

        # Statistics
        self.interrupts_raised = 0
        self.interrupts_handled = 0
        self.spurious = 0

    def raise_irq(self, irq: int):
        """Raise an interrupt request.

        Sets the corresponding bit in the IRR. The interrupt will be
        dispatched on the next call to dispatch().
        """
        if 0 <= irq < self.num_irqs:
            self.irr[irq] = True
            self.interrupts_raised += 1
        else:
            logger.warning(f"[GIC] Invalid IRQ {irq}")

    def clear_irq(self, irq: int):
        """Clear an interrupt (mark as handled)."""
        if 0 <= irq < self.num_irqs:
            self.irr[irq] = False
            self.isr[irq] = False

    def mask_irq(self, irq: int):
        """Mask (disable) an interrupt."""
        if 0 <= irq < self.num_irqs:
            self.imr[irq] = True

    def unmask_irq(self, irq: int):
        """Unmask (enable) an interrupt."""
        if 0 <= irq < self.num_irqs:
            self.imr[irq] = False

    def register_handler(self, irq: int, handler: InterruptHandler):
        """Register an interrupt handler for an IRQ."""
        self._handlers[irq] = handler

    def pending(self) -> torch.Tensor:
        """Get pending interrupts (raised, not masked, not in-service)."""
        return self.irr & ~self.imr & ~self.isr

    def dispatch(self) -> Optional[int]:
        """Dispatch the highest-priority pending interrupt.

        Returns the IRQ number that was dispatched, or None if no
        interrupts are pending.
        """
        return self.dispatch_with_policy()

    def score_pending(self, pending: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return priority scores for the currently pending interrupts."""
        if pending is None:
            pending = self.pending()

        state = torch.cat([
            self.irr.float(),
            self.isr.float(),
            self.imr.float(),
        ])

        with torch.no_grad():
            scores = self.encoder(state)

        scores = scores.clone()
        scores[~pending] = float('-inf')
        return scores

    def dispatch_with_policy(self, use_neural: Optional[bool] = None) -> Optional[int]:
        """Dispatch the highest-priority pending interrupt.

        Args:
            use_neural:
                - None: use the controller's default policy (`self._trained`)
                - True: force the neural encoder when trained
                - False: force fixed-priority dispatch
        """
        pending = self.pending()
        if not pending.any():
            return None

        if use_neural is None:
            use_neural = self._trained

        if use_neural and self._trained:
            irq = self._neural_dispatch(pending)
        else:
            irq = self._fixed_dispatch(pending)

        if irq is None:
            return None

        # Mark in-service and clear request
        self.isr[irq] = True
        self.irr[irq] = False

        # Invoke handler
        handler = self._handlers.get(irq)
        if handler is not None:
            handler(irq)
            self.interrupts_handled += 1
        else:
            self.spurious += 1
            logger.debug(f"[GIC] Spurious IRQ {irq} — no handler")

        # Clear in-service
        self.isr[irq] = False

        return irq

    def dispatch_all(self, use_neural: Optional[bool] = None) -> List[int]:
        """Dispatch all pending interrupts in priority order."""
        dispatched = []
        while True:
            irq = self.dispatch_with_policy(use_neural=use_neural)
            if irq is None:
                break
            dispatched.append(irq)
        return dispatched

    def _fixed_dispatch(self, pending: torch.Tensor) -> Optional[int]:
        """Fixed priority dispatch: lowest IRQ number wins."""
        indices = pending.nonzero(as_tuple=True)[0]
        if len(indices) == 0:
            return None
        return int(indices[0].item())

    def _neural_dispatch(self, pending: torch.Tensor) -> Optional[int]:
        """Neural priority dispatch: learned priority ordering."""
        scores = self.score_pending(pending)
        if (scores == float('-inf')).all():
            return None

        return int(scores.argmax().item())

    # ─── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str = "models/research/neuros/gic.pt"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.encoder.state_dict(), path)

    def load(self, path: str = "models/research/neuros/gic.pt") -> bool:
        if not Path(path).exists():
            return False
        self.encoder.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True))
        self.encoder.eval()
        self._trained = True
        return True

    # ─── Diagnostics ──────────────────────────────────────────────────────

    def stats(self) -> Dict:
        return {
            "raised": self.interrupts_raised,
            "handled": self.interrupts_handled,
            "spurious": self.spurious,
            "pending": int(self.pending().sum().item()),
            "in_service": int(self.isr.sum().item()),
            "masked": int(self.imr.sum().item()),
            "trained": self._trained,
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (f"NeuralGIC(pending={s['pending']}, "
                f"handled={s['handled']}, "
                f"policy={'neural' if s['trained'] else 'fixed'})")
