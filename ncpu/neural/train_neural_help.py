#!/usr/bin/env python3
"""Train the Neural Help Retrieval model.

Replaces the static dictionary lookup in NeuralHelpGenerator with a real
contrastive-learning retrieval model. The model encodes queries and help
entries into a shared embedding space; at inference time, the best-matching
help entry is retrieved via cosine similarity.

Architecture:
    char_embed(256, 32) -> LSTM(32, 64) -> L2-normalized embedding

Training: InfoNCE contrastive loss. For each query, the correct help entry
is the positive and all other entries in the batch are negatives.

Target: >85% Recall@1 (correct help entry ranked first).
"""

import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# ── Help entries (command -> help text) ───────────────────────────────────

HELP_ENTRIES = {
    "ls": "ls: List directory contents. Try: ls -la, ls /home, ls | grep .c",
    "cat": "cat: Display file contents. Try: cat file.txt, cat /etc/motd",
    "echo": "echo: Print text. Try: echo hello > file.txt, echo $HOME",
    "cc": "cc: Compile C source. Try: cc hello.c && run /bin/hello",
    "run": "run: Execute a compiled binary. Try: run /bin/hello",
    "grep": "grep: Search in files. Try: ls | grep .c, grep pattern file.txt",
    "wc": "wc: Count words/lines. Try: wc file.txt, cat file | wc",
    "sort": "sort: Sort lines. Try: ls | sort, sort -r file.txt",
    "ps": "ps: Show processes. Try: ps (in --multiproc mode)",
    "cd": "cd: Change directory. Try: cd /home/user, cd .., cd /tmp",
    "pwd": "pwd: Print working directory.",
    "mkdir": "mkdir: Create directory. Try: mkdir /tmp/mydir",
    "rm": "rm: Remove file. Try: rm /tmp/test.txt",
    "touch": "touch: Create empty file. Try: touch /tmp/new.txt",
    "cp": "cp: Copy file. Try: cp src.txt dst.txt",
    "head": "head: Show first lines. Try: head file.txt",
    "tee": "tee: Write to file and stdout. Try: ls | tee /tmp/listing.txt",
    "uniq": "uniq: Remove duplicates. Try: sort file.txt | uniq",
    "exit": "exit: Exit the shell.",
    "help": "help: Show available commands and contextual tips.",
    "kill": "kill: Send signal to process. Try: kill <pid>",
    "env": "env: Show environment variables.",
    "export": "export: Set environment variable. Try: export VAR=value",
}

HELP_INDEX = list(HELP_ENTRIES.keys())
HELP_TEXTS = [HELP_ENTRIES[k] for k in HELP_INDEX]

# ── Query generation ──────────────────────────────────────────────────────

QUERY_TEMPLATES = {
    "ls": [
        "how to list files", "show directory contents", "list files in folder",
        "what's in this directory", "ls command", "how do I see files",
        "show me the files", "directory listing", "list", "list directory",
        "view folder contents", "what files are here", "dir listing",
    ],
    "cat": [
        "how to read a file", "display file contents", "show file text",
        "cat command", "view file", "read file contents", "print file",
        "show what's in a file", "open file", "cat", "display text file",
    ],
    "echo": [
        "how to print text", "write to a file", "create a file with text",
        "echo command", "output text", "print string", "write text to file",
        "echo hello", "redirect output", "print", "display text",
    ],
    "cc": [
        "how to compile", "compile C code", "build program", "cc command",
        "compile source", "make binary", "build C file", "compiler",
        "how to build", "compile hello.c", "create executable",
    ],
    "run": [
        "how to run a program", "execute binary", "run command",
        "start program", "execute compiled", "launch binary",
        "run /bin/hello", "execute", "how to execute",
    ],
    "grep": [
        "how to search", "find text in file", "search pattern",
        "grep command", "filter output", "search files", "find string",
        "pattern match", "grep", "search for text",
    ],
    "wc": [
        "count lines", "count words", "how many lines", "wc command",
        "word count", "line count", "count characters", "wc",
    ],
    "sort": [
        "sort lines", "sort output", "alphabetical order", "sort command",
        "sort file", "order lines", "sort", "arrange alphabetically",
    ],
    "ps": [
        "show processes", "list processes", "process status", "ps command",
        "running processes", "what's running", "ps", "process list",
    ],
    "cd": [
        "change directory", "go to folder", "navigate", "cd command",
        "switch directory", "move to", "cd", "change folder",
    ],
    "pwd": [
        "current directory", "where am I", "print working directory",
        "pwd command", "show path", "current path", "pwd",
    ],
    "mkdir": [
        "create directory", "make folder", "new directory", "mkdir command",
        "create folder", "mkdir", "make directory",
    ],
    "rm": [
        "delete file", "remove file", "rm command", "erase file",
        "delete", "remove", "rm", "how to delete",
    ],
    "touch": [
        "create empty file", "touch command", "new file", "make file",
        "create file", "touch", "empty file",
    ],
    "cp": [
        "copy file", "duplicate file", "cp command", "copy",
        "make a copy", "cp", "how to copy",
    ],
    "head": [
        "show first lines", "beginning of file", "head command",
        "first few lines", "top of file", "head", "show start",
    ],
    "tee": [
        "write to file and screen", "tee command", "split output",
        "save and display", "tee", "redirect and show",
    ],
    "uniq": [
        "remove duplicates", "unique lines", "uniq command",
        "deduplicate", "uniq", "filter duplicates",
    ],
    "exit": [
        "quit shell", "exit command", "leave", "close shell",
        "exit", "quit", "how to exit",
    ],
    "help": [
        "show help", "available commands", "what can I do",
        "help command", "how to use", "help", "commands list",
    ],
    "kill": [
        "kill process", "terminate process", "send signal",
        "kill command", "stop process", "kill",
    ],
    "env": [
        "environment variables", "show env", "env command",
        "variables", "env", "show environment",
    ],
    "export": [
        "set variable", "export variable", "export command",
        "define variable", "export", "set env",
    ],
}


def add_query_noise(text: str) -> str:
    """Add noise to query strings for augmentation."""
    r = random.random()
    if r < 0.1:
        return text.upper()
    elif r < 0.15:
        # Typo: swap two adjacent chars
        if len(text) > 3:
            idx = random.randint(1, len(text) - 2)
            chars = list(text)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            return "".join(chars)
    elif r < 0.2:
        return "how do i " + text
    elif r < 0.25:
        return text + "?"
    return text


def generate_dataset(pairs_per_command: int = 80):
    """Generate (query, command_index) pairs."""
    dataset = []
    for cmd_idx, cmd in enumerate(HELP_INDEX):
        templates = QUERY_TEMPLATES.get(cmd, [cmd, f"how to {cmd}", f"{cmd} usage"])
        for _ in range(pairs_per_command):
            query = random.choice(templates)
            query = add_query_noise(query)
            dataset.append((query, cmd_idx))
    random.shuffle(dataset)
    return dataset


# ── Model ─────────────────────────────────────────────────────────────────

class NeuralHelpModel(nn.Module):
    """Retrieval-based help: encodes queries and help entries, returns best match.

    Uses shared character-level LSTM encoder. Queries and help texts are
    encoded into the same embedding space. Retrieval is by cosine similarity.
    """

    def __init__(self, vocab_size=256, embed_dim=32, hidden=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden = hidden

        self.char_embed = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden, batch_first=True, num_layers=2,
                               dropout=0.1)
        self.projection = nn.Linear(hidden, hidden)

    def encode(self, chars: torch.Tensor) -> torch.Tensor:
        """Encode character sequences to L2-normalized embeddings.

        Args:
            chars: (batch, seq_len) of byte values 0-255
        Returns:
            (batch, hidden) L2-normalized embeddings
        """
        embedded = self.char_embed(chars)
        _, (h, _) = self.encoder(embedded)
        proj = self.projection(h[-1])
        return F.normalize(proj, p=2, dim=-1)

    def forward(self, query_chars: torch.Tensor,
                help_chars: torch.Tensor) -> torch.Tensor:
        """Compute similarity between queries and help entries.

        Args:
            query_chars: (batch, seq_len) queries
            help_chars: (n_help, seq_len) all help entries
        Returns:
            (batch, n_help) cosine similarity scores
        """
        q = self.encode(query_chars)   # (batch, hidden)
        h = self.encode(help_chars)    # (n_help, hidden)
        return q @ h.T                 # (batch, n_help)

    def retrieve(self, query: str, help_texts: list[str],
                 max_len: int = 128) -> tuple[int, float]:
        """Retrieve the best-matching help entry for a query.

        Returns (index, similarity_score).
        """
        q = self._encode_text(query, max_len).unsqueeze(0)
        h = torch.stack([self._encode_text(t, max_len) for t in help_texts])
        device = next(self.parameters()).device
        q, h = q.to(device), h.to(device)
        with torch.no_grad():
            scores = self.forward(q, h)
        idx = int(scores.argmax(dim=-1).item())
        score = float(scores[0, idx].item())
        return idx, score

    @staticmethod
    def _encode_text(text: str, max_len: int = 128) -> torch.Tensor:
        """Encode a string to byte tensor."""
        codes = [min(b, 255) for b in text.encode("utf-8", errors="replace")[:max_len]]
        if len(codes) < max_len:
            codes += [0] * (max_len - len(codes))
        return torch.tensor(codes, dtype=torch.long)


# ── Training ──────────────────────────────────────────────────────────────

def train(epochs: int = 100, batch_size: int = 64, lr: float = 1e-3,
          pairs_per_command: int = 80, temperature: float = 0.07):
    """Train the help retrieval model with InfoNCE contrastive loss."""
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"

    print(f"Training NeuralHelpModel on {device}")
    n_commands = len(HELP_INDEX)
    print(f"  {n_commands} help entries, {pairs_per_command} queries/entry")
    print(f"  {epochs} epochs, batch_size={batch_size}, lr={lr}, temp={temperature}")

    # Generate data
    dataset = generate_dataset(pairs_per_command)
    n = len(dataset)
    split = int(0.85 * n)
    train_data = dataset[:split]
    val_data = dataset[split:]
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    max_len = 128

    # Pre-encode all help texts
    help_chars = torch.stack([
        NeuralHelpModel._encode_text(t, max_len) for t in HELP_TEXTS
    ]).to(device)

    model = NeuralHelpModel().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_r1 = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        random.shuffle(train_data)
        total_loss = 0.0
        correct = 0
        total = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            queries, labels = zip(*batch)
            q_chars = torch.stack([
                NeuralHelpModel._encode_text(q, max_len) for q in queries
            ]).to(device)
            labels_t = torch.tensor(labels, dtype=torch.long, device=device)

            # InfoNCE: query vs all help entries
            scores = model(q_chars, help_chars) / temperature
            loss = F.cross_entropy(scores, labels_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)
            correct += (scores.argmax(dim=-1) == labels_t).sum().item()
            total += len(batch)

        scheduler.step()
        train_r1 = correct / max(total, 1)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i + batch_size]
                queries, labels = zip(*batch)
                q_chars = torch.stack([
                    NeuralHelpModel._encode_text(q, max_len) for q in queries
                ]).to(device)
                labels_t = torch.tensor(labels, dtype=torch.long, device=device)
                scores = model(q_chars, help_chars) / temperature
                val_correct += (scores.argmax(dim=-1) == labels_t).sum().item()
                val_total += len(batch)
        val_r1 = val_correct / max(val_total, 1)

        if val_r1 > best_val_r1:
            best_val_r1 = val_r1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"loss={total_loss / total:.4f}, "
                  f"train_R@1={train_r1:.1%}, val_R@1={val_r1:.1%}")

    print(f"\n  Best validation Recall@1: {best_val_r1:.1%}")

    # Save
    save_path = PROJECT_ROOT / "models" / "os" / "neural_help.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, str(save_path))
    print(f"  Saved to {save_path}")

    # Per-command accuracy
    model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()
    print("\n  Per-command Recall@1:")
    for cmd_idx, cmd in enumerate(HELP_INDEX):
        class_data = [(q, l) for q, l in dataset if l == cmd_idx]
        if not class_data:
            continue
        queries, labels = zip(*class_data)
        q_chars = torch.stack([
            NeuralHelpModel._encode_text(q, max_len) for q in queries
        ]).to(device)
        labels_t = torch.tensor(labels, dtype=torch.long, device=device)
        with torch.no_grad():
            scores = model(q_chars, help_chars)
            acc = (scores.argmax(dim=-1) == labels_t).float().mean().item()
        print(f"    {cmd:10s}: {acc:.1%}")

    return best_val_r1


if __name__ == "__main__":
    r1 = train()
    if r1 >= 0.85:
        print(f"\n  Target met: {r1:.1%} >= 85%")
    else:
        print(f"\n  Below target ({r1:.1%} < 85%), retraining with more data...")
        r1 = train(epochs=150, pairs_per_command=150)
        print(f"  Final Recall@1: {r1:.1%}")
