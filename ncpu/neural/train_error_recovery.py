#!/usr/bin/env python3
"""Train the Neural Error Recovery model.

Replaces the pattern-matching dictionary in neural_demo.py with a real
character-level LSTM classifier that maps error text to recovery suggestion
indices.

Architecture:
    char_embed(256, 32) -> LSTM(32, 64) -> Linear(64, 12)

Training data: 12 error categories, each augmented with ~80 variations
(typos, different file names, partial matches, case variations, word order).

Target: >80% accuracy on the 12 error categories.
"""

import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# ── Error categories and their suggestion indices ─────────────────────────

ERROR_CATEGORIES = {
    0: {
        "pattern": "not found",
        "suggestion": "Check spelling or use 'ls' to list available files",
        "templates": [
            "{name}: not found",
            "command not found: {name}",
            "{name} not found in /bin",
            "file not found: {path}",
            "{path}: No such command not found",
            "binary not found at {path}",
            "not found: '{name}'",
            "error: not found",
        ],
    },
    1: {
        "pattern": "permission denied",
        "suggestion": "File may be read-only",
        "templates": [
            "permission denied: {path}",
            "{path}: permission denied",
            "cannot write to {path}: permission denied",
            "access denied: {path}",
            "Permission Denied accessing {path}",
            "{name}: Permission Denied",
            "error: permission denied for {path}",
            "write permission denied: {path}",
        ],
    },
    2: {
        "pattern": "no such file",
        "suggestion": "Use 'ls' to check the path exists",
        "templates": [
            "no such file or directory: {path}",
            "{path}: no such file",
            "cannot open '{path}': no such file or directory",
            "no such file: {path}",
            "error: no such file '{name}'",
            "{name}: No such file or directory",
            "open failed: no such file {path}",
            "stat: no such file: {path}",
        ],
    },
    3: {
        "pattern": "syntax error",
        "suggestion": "Check command syntax with 'help'",
        "templates": [
            "syntax error near '{token}'",
            "syntax error: unexpected '{token}'",
            "parse error: syntax error at line {n}",
            "syntax error in expression",
            "{name}: syntax error",
            "error: syntax error before '{token}'",
            "unexpected token: syntax error",
            "syntax error: missing ';'",
        ],
    },
    4: {
        "pattern": "compilation failed",
        "suggestion": "Check C source for errors",
        "templates": [
            "compilation failed: {path}",
            "cc: compilation failed for {name}",
            "error: compilation failed with {n} errors",
            "compilation failed",
            "{name}.c: compilation failed",
            "compile error: compilation failed",
            "build failed: compilation failed",
            "cc: failed to compile {name}",
        ],
    },
    5: {
        "pattern": "unknown command",
        "suggestion": "Type 'help' to list available commands",
        "templates": [
            "unknown command: {name}",
            "{name}: unknown command",
            "error: unknown command '{name}'",
            "shell: unknown command {name}",
            "unknown command",
            "'{name}' is not a recognized command, unknown command",
            "command unknown: {name}",
            "unrecognized command: unknown command {name}",
        ],
    },
    6: {
        "pattern": "cannot open",
        "suggestion": "Verify the file path with 'ls'",
        "templates": [
            "cannot open '{path}'",
            "cannot open file: {path}",
            "error: cannot open {path} for reading",
            "{name}: cannot open",
            "failed: cannot open {path}",
            "cannot open '{name}': file not accessible",
            "open: cannot open {path}",
            "error: cannot open source file {path}",
        ],
    },
    7: {
        "pattern": "invalid argument",
        "suggestion": "Check argument types and ranges",
        "templates": [
            "invalid argument: {name}",
            "{name}: invalid argument",
            "error: invalid argument '{token}'",
            "invalid argument to {name}",
            "invalid argument: expected number, got '{token}'",
            "invalid argument for option '{name}'",
            "{name}: invalid argument value",
            "argument error: invalid argument",
        ],
    },
    8: {
        "pattern": "directory not empty",
        "suggestion": "Use 'rm' on contents first, then 'rmdir'",
        "templates": [
            "directory not empty: {path}",
            "rmdir: directory not empty",
            "{path}: directory not empty",
            "cannot remove '{path}': directory not empty",
            "error: directory not empty at {path}",
            "rmdir failed: directory not empty",
            "remove: directory not empty: {path}",
            "directory not empty, cannot remove",
        ],
    },
    9: {
        "pattern": "already exists",
        "suggestion": "File or directory already exists at that path",
        "templates": [
            "already exists: {path}",
            "{path}: already exists",
            "mkdir: '{path}' already exists",
            "file already exists: {path}",
            "cannot create '{path}': already exists",
            "error: {name} already exists",
            "directory already exists at {path}",
            "target already exists: {path}",
        ],
    },
    10: {
        "pattern": "segmentation fault",
        "suggestion": "Memory access violation -- check array bounds",
        "templates": [
            "segmentation fault",
            "segmentation fault (core dumped)",
            "SIGSEGV: segmentation fault at address {addr}",
            "segmentation fault in {name}",
            "fatal: segmentation fault",
            "program terminated: segmentation fault",
            "signal 11: segmentation fault",
            "segfault: segmentation fault at {addr}",
        ],
    },
    11: {
        "pattern": "stack overflow",
        "suggestion": "Recursive call depth exceeded -- add base case",
        "templates": [
            "stack overflow",
            "stack overflow detected",
            "fatal: stack overflow in {name}",
            "stack overflow at depth {n}",
            "error: stack overflow",
            "program aborted: stack overflow",
            "maximum recursion depth exceeded: stack overflow",
            "signal: stack overflow",
        ],
    },
}

# ── Substitution pools ────────────────────────────────────────────────────

NAMES = [
    "foo", "bar", "test", "main", "hello", "world", "fib", "sieve",
    "sort", "grep", "calc", "prog", "app", "run", "build", "cc",
    "myfile", "data", "config", "init", "setup", "util",
]

PATHS = [
    "/home/user/test.c", "/tmp/test.txt", "/bin/hello", "/var/log/syslog",
    "/home/user/foo.c", "/etc/config", "/usr/lib/libc.so", "/home/user/data.txt",
    "/tmp/results/a.txt", "/home/user/prog.c", "/bin/sieve", "/home/user/sort.c",
    "/tmp/output.txt", "/home/user/README.txt", "/etc/motd",
]

TOKENS = [
    "}", "{", ";", ")", "(", "=", "+", "if", "else", "return",
    "int", "void", "for", "while", "EOF", "|", "&&",
]

ADDRS = ["0x0", "0x4000", "0xDEAD", "0xFFFF0000", "0x10000", "0x50000"]
NUMBERS = ["0", "1", "2", "3", "5", "10", "42", "100", "256"]


def fill_template(template: str) -> str:
    """Fill a template with random substitutions."""
    result = template
    result = result.replace("{name}", random.choice(NAMES))
    result = result.replace("{path}", random.choice(PATHS))
    result = result.replace("{token}", random.choice(TOKENS))
    result = result.replace("{addr}", random.choice(ADDRS))
    result = result.replace("{n}", random.choice(NUMBERS))
    return result


def add_noise(text: str) -> str:
    """Add realistic noise: case changes, extra spaces, truncation."""
    noise_type = random.random()
    if noise_type < 0.15:
        # Random case change
        return text.upper() if random.random() < 0.5 else text.lower()
    elif noise_type < 0.25:
        # Extra whitespace
        words = text.split()
        idx = random.randint(0, max(0, len(words) - 1))
        words[idx] = "  " + words[idx] + "  "
        return " ".join(words)
    elif noise_type < 0.35:
        # Prefix with noise
        prefixes = ["error: ", "fatal: ", "[err] ", ">> ", "!! "]
        return random.choice(prefixes) + text
    elif noise_type < 0.42:
        # Truncate to partial match
        cut = max(10, int(len(text) * random.uniform(0.5, 0.9)))
        return text[:cut]
    return text


def generate_dataset(samples_per_class: int = 200) -> list:
    """Generate (text, label) pairs for all 12 error categories."""
    dataset = []
    for label, info in ERROR_CATEGORIES.items():
        for _ in range(samples_per_class):
            template = random.choice(info["templates"])
            text = fill_template(template)
            # 40% chance of noise augmentation
            if random.random() < 0.4:
                text = add_noise(text)
            dataset.append((text, label))
    random.shuffle(dataset)
    return dataset


# ── Model ─────────────────────────────────────────────────────────────────

class NeuralErrorRecoveryModel(nn.Module):
    """Classifies error messages and suggests recovery actions.

    Character-level LSTM encoder maps error text to one of 12
    suggestion categories. Operates on raw byte values (0-255).
    """

    def __init__(self, vocab_size=256, embed_dim=32, hidden=64, n_suggestions=12):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.n_suggestions = n_suggestions

        self.char_embed = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden, batch_first=True, num_layers=2,
                               dropout=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, n_suggestions),
        )

    def forward(self, error_chars: torch.Tensor) -> torch.Tensor:
        """
        Args:
            error_chars: (batch, seq_len) of byte values 0-255
        Returns:
            (batch, n_suggestions) logits
        """
        embedded = self.char_embed(error_chars)
        _, (h, _) = self.encoder(embedded)
        # Use last layer's hidden state
        return self.classifier(h[-1])

    def predict(self, text: str) -> int:
        """Predict the suggestion index for an error string."""
        chars = self._encode_text(text)
        with torch.no_grad():
            logits = self.forward(chars.unsqueeze(0))
        return int(logits.argmax(dim=-1).item())

    @staticmethod
    def _encode_text(text: str, max_len: int = 128) -> torch.Tensor:
        """Encode a string to byte tensor, padded/truncated to max_len."""
        codes = [min(b, 255) for b in text.encode("utf-8", errors="replace")[:max_len]]
        if len(codes) < max_len:
            codes += [0] * (max_len - len(codes))
        return torch.tensor(codes, dtype=torch.long)


# ── Training ──────────────────────────────────────────────────────────────

def train(epochs: int = 80, batch_size: int = 64, lr: float = 1e-3,
          samples_per_class: int = 200):
    """Train the error recovery model and save to models/os/error_recovery.pt."""
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    print(f"Training NeuralErrorRecoveryModel on {device}")
    print(f"  {12} error categories, {samples_per_class} samples/class")
    print(f"  {epochs} epochs, batch_size={batch_size}, lr={lr}")

    # Generate data
    dataset = generate_dataset(samples_per_class)
    n = len(dataset)
    split = int(0.85 * n)
    train_data = dataset[:split]
    val_data = dataset[split:]
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    # Encode
    max_len = 128

    def encode_batch(data):
        texts, labels = zip(*data)
        chars = torch.stack([NeuralErrorRecoveryModel._encode_text(t, max_len) for t in texts])
        labels_t = torch.tensor(labels, dtype=torch.long)
        return chars.to(device), labels_t.to(device)

    model = NeuralErrorRecoveryModel().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        random.shuffle(train_data)
        total_loss = 0.0
        correct = 0
        total = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            chars, labels = encode_batch(batch)

            logits = model(chars)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += len(batch)

        scheduler.step()
        train_acc = correct / max(total, 1)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i + batch_size]
                chars, labels = encode_batch(batch)
                logits = model(chars)
                val_correct += (logits.argmax(dim=-1) == labels).sum().item()
                val_total += len(batch)
        val_acc = val_correct / max(val_total, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"loss={total_loss / total:.4f}, "
                  f"train_acc={train_acc:.1%}, val_acc={val_acc:.1%}")

    print(f"\n  Best validation accuracy: {best_val_acc:.1%}")

    # Save
    save_path = PROJECT_ROOT / "models" / "os" / "error_recovery.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, str(save_path))
    print(f"  Saved to {save_path}")

    # Per-class accuracy on full dataset
    model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()
    print("\n  Per-class accuracy:")
    for label, info in ERROR_CATEGORIES.items():
        class_data = [(t, l) for t, l in dataset if l == label]
        chars, labels = encode_batch(class_data)
        with torch.no_grad():
            logits = model(chars)
            acc = (logits.argmax(dim=-1) == labels).float().mean().item()
        print(f"    [{label:2d}] {info['pattern']:20s}: {acc:.1%}")

    return best_val_acc


# ── Suggestion table (matches ERROR_CATEGORIES indices) ───────────────────

SUGGESTIONS = [info["suggestion"] for _, info in sorted(ERROR_CATEGORIES.items())]


if __name__ == "__main__":
    acc = train()
    target = 0.80
    if acc >= target:
        print(f"\n  Target met: {acc:.1%} >= {target:.0%}")
    else:
        print(f"\n  WARNING: {acc:.1%} < {target:.0%} target, retraining with more data...")
        acc = train(epochs=120, samples_per_class=400)
        print(f"  Final accuracy: {acc:.1%}")
