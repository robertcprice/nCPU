#!/usr/bin/env python3
"""
Training script for Neural Dynamic Linker components.

Trains:
1. NeuralSymbolResolver - Symbol name matching with attention
2. NeuralRelocationEngine - Relocation type routing (MoE)

Training data sources:
- Real ARM64 ELF binaries (BusyBox, Alpine packages)
- Synthetic symbol pairs with known matches
- ld.so execution traces for validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import os
import struct
import random
import time
from pathlib import Path

from neural_dynamic_linker import (
    NeuralSymbolResolver,
    NeuralRelocationEngine,
    Symbol,
    generate_symbol_training_data
)


# ════════════════════════════════════════════════════════════════════════════════
# TRAINING DATASETS
# ════════════════════════════════════════════════════════════════════════════════

class SymbolMatchDataset(Dataset):
    """
    Dataset of symbol name pairs for training the resolver.

    Contains:
    - Positive pairs: undefined name + matching definition
    - Negative pairs: undefined name + non-matching definition
    """

    def __init__(
        self,
        num_samples: int = 50000,
        from_elfs: bool = True,
        sysroot: str = ""
    ):
        self.queries = []
        self.keys = []
        self.labels = []

        # Generate synthetic data
        print("  Generating synthetic symbol pairs...")
        syn_q, syn_k, syn_l = self._generate_synthetic(num_samples // 2)
        self.queries.extend(syn_q)
        self.keys.extend(syn_k)
        self.labels.extend(syn_l.tolist())

        # Extract from real ELFs if sysroot available
        if from_elfs and sysroot and os.path.isdir(sysroot):
            print(f"  Extracting symbols from sysroot: {sysroot}")
            elf_q, elf_k, elf_l = self._extract_from_sysroot(sysroot, num_samples // 2)
            self.queries.extend(elf_q)
            self.keys.extend(elf_k)
            self.labels.extend(elf_l)

        print(f"  Total samples: {len(self.queries)}")

    def _generate_synthetic(self, num_samples: int) -> Tuple[List[str], List[str], torch.Tensor]:
        """Generate synthetic symbol matching pairs."""
        # Common C library functions
        base_functions = [
            # Memory
            'malloc', 'calloc', 'realloc', 'free', 'mmap', 'munmap',
            'memcpy', 'memmove', 'memset', 'memcmp',
            # String
            'strlen', 'strcpy', 'strncpy', 'strcmp', 'strncmp',
            'strcat', 'strncat', 'strchr', 'strrchr', 'strstr',
            # I/O
            'printf', 'fprintf', 'sprintf', 'snprintf',
            'scanf', 'fscanf', 'sscanf',
            'fopen', 'fclose', 'fread', 'fwrite', 'fseek', 'ftell',
            'open', 'close', 'read', 'write', 'lseek',
            # Process
            'fork', 'exec', 'execve', 'exit', '_exit', 'abort',
            'wait', 'waitpid', 'kill', 'signal', 'sigaction',
            # Thread
            'pthread_create', 'pthread_join', 'pthread_exit',
            'pthread_mutex_init', 'pthread_mutex_lock', 'pthread_mutex_unlock',
            'pthread_cond_init', 'pthread_cond_wait', 'pthread_cond_signal',
            # Dynamic
            'dlopen', 'dlclose', 'dlsym', 'dlerror',
            # Misc
            'getenv', 'setenv', 'getpid', 'getuid', 'getgid',
            'time', 'gettimeofday', 'sleep', 'usleep', 'nanosleep',
        ]

        prefixes = ['', '_', '__', '___']
        suffixes = ['', '_r', '_l', '_internal', '_impl', '_default']
        versions = ['', '@@GLIBC_2.17', '@@GLIBC_2.4', '@@GLIBC_2.2.5', '@GLIBC_2.0']

        queries = []
        keys = []
        labels = []

        for _ in range(num_samples):
            base = random.choice(base_functions)

            # Positive pair: same base, possibly different prefix/suffix/version
            if random.random() < 0.7:  # 70% positive
                q_prefix = random.choice(prefixes)
                k_prefix = random.choice(prefixes)
                k_suffix = random.choice(suffixes) if random.random() < 0.3 else ''
                k_version = random.choice(versions)

                queries.append(q_prefix + base)
                keys.append(k_prefix + base + k_suffix + k_version)
                labels.append(1.0)
            else:  # 30% negative
                other_base = random.choice(base_functions)
                while other_base == base:
                    other_base = random.choice(base_functions)

                q_prefix = random.choice(prefixes)
                k_prefix = random.choice(prefixes)
                k_version = random.choice(versions)

                queries.append(q_prefix + base)
                keys.append(k_prefix + other_base + k_version)
                labels.append(0.0)

        return queries, keys, torch.tensor(labels)

    def _extract_from_sysroot(
        self,
        sysroot: str,
        max_samples: int
    ) -> Tuple[List[str], List[str], List[float]]:
        """Extract symbol pairs from real ELF files."""
        queries = []
        keys = []
        labels = []

        # Find shared libraries
        lib_dirs = ['lib', 'lib64', 'usr/lib', 'usr/lib64']
        libraries = []

        for lib_dir in lib_dirs:
            path = Path(sysroot) / lib_dir
            if path.exists():
                for f in path.glob('*.so*'):
                    if f.is_file():
                        libraries.append(str(f))

        if not libraries:
            return queries, keys, labels

        # Extract symbols from libraries
        all_symbols = set()
        for lib_path in libraries[:20]:  # Limit to avoid OOM
            try:
                with open(lib_path, 'rb') as f:
                    data = f.read()
                symbols = self._extract_elf_symbols(data)
                all_symbols.update(symbols)
            except Exception:
                pass

        if not all_symbols:
            return queries, keys, labels

        symbol_list = list(all_symbols)
        samples_per_symbol = max(1, max_samples // len(symbol_list))

        for sym in symbol_list:
            # Create positive pairs (exact match)
            queries.append(sym)
            keys.append(sym)
            labels.append(1.0)

            # Create positive pairs (with version stripping)
            if '@@' in sym:
                base = sym.split('@@')[0]
                queries.append(base)
                keys.append(sym)
                labels.append(1.0)

            # Create negative pairs
            if len(symbol_list) > 1:
                other = random.choice(symbol_list)
                while other == sym:
                    other = random.choice(symbol_list)
                queries.append(sym.split('@@')[0] if '@@' in sym else sym)
                keys.append(other)
                labels.append(0.0)

            if len(queries) >= max_samples:
                break

        return queries, keys, labels

    def _extract_elf_symbols(self, data: bytes) -> List[str]:
        """Extract symbol names from ELF file."""
        symbols = []

        if len(data) < 64 or data[:4] != b'\x7fELF':
            return symbols

        try:
            # Parse section headers
            e_shoff = struct.unpack('<Q', data[40:48])[0]
            e_shentsize = struct.unpack('<H', data[58:60])[0]
            e_shnum = struct.unpack('<H', data[60:62])[0]
            e_shstrndx = struct.unpack('<H', data[62:64])[0]

            # Find .dynsym and .dynstr
            for i in range(e_shnum):
                sh_offset = e_shoff + i * e_shentsize
                if sh_offset + 64 > len(data):
                    break

                sh_type = struct.unpack('<I', data[sh_offset+4:sh_offset+8])[0]

                # SHT_DYNSYM = 11
                if sh_type == 11:
                    sh_off = struct.unpack('<Q', data[sh_offset+24:sh_offset+32])[0]
                    sh_size = struct.unpack('<Q', data[sh_offset+32:sh_offset+40])[0]
                    sh_link = struct.unpack('<I', data[sh_offset+40:sh_offset+44])[0]
                    sh_entsize = struct.unpack('<Q', data[sh_offset+56:sh_offset+64])[0]

                    if sh_entsize == 0:
                        sh_entsize = 24  # Default for 64-bit

                    # Get string table
                    strtab_hdr = e_shoff + sh_link * e_shentsize
                    strtab_off = struct.unpack('<Q', data[strtab_hdr+24:strtab_hdr+32])[0]
                    strtab_size = struct.unpack('<Q', data[strtab_hdr+32:strtab_hdr+40])[0]

                    # Extract symbol names
                    num_syms = sh_size // sh_entsize
                    for j in range(min(num_syms, 500)):  # Limit
                        sym_off = sh_off + j * sh_entsize
                        if sym_off + 24 > len(data):
                            break

                        st_name = struct.unpack('<I', data[sym_off:sym_off+4])[0]
                        st_info = data[sym_off + 4]

                        # Check if global or weak
                        binding = st_info >> 4
                        if binding > 0:  # STB_GLOBAL or STB_WEAK
                            str_off = strtab_off + st_name
                            if str_off < len(data):
                                str_end = data.find(b'\x00', str_off)
                                if str_end > str_off:
                                    name = data[str_off:str_end].decode('utf-8', errors='ignore')
                                    if name and len(name) < 128:
                                        symbols.append(name)
                    break
        except Exception:
            pass

        return symbols

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.keys[idx], self.labels[idx]


def collate_fn(batch):
    """Custom collate function for symbol batches."""
    queries, keys, labels = zip(*batch)
    return list(queries), list(keys), torch.tensor(labels)


# ════════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════════

def train_symbol_resolver(
    resolver: NeuralSymbolResolver,
    train_dataset: SymbolMatchDataset,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    save_path: str = "symbol_resolver.pt"
):
    """
    Train the NeuralSymbolResolver on symbol matching.

    Uses contrastive learning: maximize similarity for positive pairs,
    minimize for negative pairs.
    """
    device = resolver.device
    optimizer = torch.optim.AdamW(resolver.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    print(f"\n{'='*60}")
    print(f"  Training NeuralSymbolResolver")
    print(f"  Samples: {len(train_dataset):,}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"{'='*60}\n")

    best_acc = 0
    for epoch in range(epochs):
        resolver.train()
        total_loss = 0
        correct = 0
        total = 0

        for queries, keys, labels in dataloader:
            labels = labels.to(device)

            # Encode symbols
            query_emb = resolver.encode_symbol_names(queries)
            key_emb = resolver.encode_symbol_names(keys)

            # Normalize for cosine similarity
            query_norm = F.normalize(query_emb, dim=-1)
            key_norm = F.normalize(key_emb, dim=-1)

            # Cosine similarity
            similarity = (query_norm * key_norm).sum(dim=-1)  # [batch]

            # Contrastive loss: MSE between similarity and labels
            # Labels: 1 for match, 0 for non-match
            loss = F.mse_loss(similarity, labels)

            # Accuracy
            predictions = (similarity > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += len(labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(resolver.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(resolver.state_dict(), save_path)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}, acc={accuracy:.1%}, best={best_acc:.1%}")

    print(f"\n  ✅ Training complete! Best accuracy: {best_acc:.1%}")
    print(f"  Model saved to: {save_path}")

    return best_acc


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Neural Dynamic Linker")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--samples", type=int, default=50000, help="Training samples")
    parser.add_argument("--sysroot", type=str, default="", help="Alpine sysroot path")
    parser.add_argument("--output", type=str, default="symbol_resolver.pt", help="Output path")
    args = parser.parse_args()

    print("=" * 70)
    print("  Neural Dynamic Linker Training")
    print("=" * 70)

    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Get sysroot from env if not provided
    sysroot = args.sysroot or os.environ.get('NEURAL_SYSROOT', '')

    # Create dataset
    print("\n📊 Creating training dataset...")
    dataset = SymbolMatchDataset(
        num_samples=args.samples,
        from_elfs=bool(sysroot),
        sysroot=sysroot
    )

    # Create resolver
    print("\n🧠 Creating NeuralSymbolResolver...")
    resolver = NeuralSymbolResolver(device=device)
    params = sum(p.numel() for p in resolver.parameters())
    print(f"  Parameters: {params:,}")

    # Train
    best_acc = train_symbol_resolver(
        resolver,
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.output
    )

    # Test
    print("\n🧪 Testing trained resolver...")
    resolver.load_state_dict(torch.load(args.output))
    resolver.eval()

    test_queries = ['malloc', 'printf', 'strlen', '__libc_start_main']
    test_keys = ['malloc', 'printf@@GLIBC_2.17', 'strlen', '__libc_start_main@@GLIBC_2.34']
    test_addrs = torch.tensor([0x1000, 0x2000, 0x3000, 0x4000], device=device)

    with torch.no_grad():
        addrs, confs = resolver.resolve(test_queries, test_keys, test_addrs)

    print("\n  Resolution test:")
    for i, (q, k) in enumerate(zip(test_queries, test_keys)):
        print(f"    {q} → {k}: addr=0x{int(addrs[i]):X}, conf={float(confs[i]):.3f}")

    print("\n" + "=" * 70)
    print(f"  Training complete! Final accuracy: {best_acc:.1%}")
    print("=" * 70)
