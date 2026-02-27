#!/usr/bin/env python3
"""
Neural Dynamic Linker - Fully Neural Symbol Resolution and Relocation

This module implements a neural approach to dynamic linking for ARM64 binaries:
1. NeuralSymbolResolver: Attention-based symbol name matching
2. SymbolTableEmbedding: Pre-computed embeddings for fast lookup
3. NeuralRelocationEngine: MoE-based relocation type handling
4. NeuralDynamicLinker: Orchestrator for the full linking process

Key insight: Symbol resolution is essentially a learned similarity matching problem
where we match undefined references to exported definitions based on name similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum
import struct


# ════════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════════

class ARM64Reloc(IntEnum):
    """ARM64 relocation types."""
    R_AARCH64_NONE = 0
    R_AARCH64_ABS64 = 257
    R_AARCH64_COPY = 1024
    R_AARCH64_GLOB_DAT = 1025
    R_AARCH64_JUMP_SLOT = 1026
    R_AARCH64_RELATIVE = 1027
    R_AARCH64_TLS_DTPREL64 = 1028
    R_AARCH64_TLS_DTPMOD64 = 1029
    R_AARCH64_TLS_TPREL64 = 1030


@dataclass
class Symbol:
    """Represents an ELF symbol."""
    name: str
    address: int
    size: int
    binding: int  # STB_LOCAL=0, STB_GLOBAL=1, STB_WEAK=2
    type: int     # STT_NOTYPE=0, STT_OBJECT=1, STT_FUNC=2
    section: int  # Section index
    library: str = ""  # Source library name


@dataclass
class Relocation:
    """Represents an ELF relocation entry."""
    offset: int       # Where to apply relocation
    type: int         # Relocation type (ARM64Reloc)
    symbol_idx: int   # Symbol table index
    addend: int       # Addend value
    symbol_name: str = ""


@dataclass
class LoadedLibrary:
    """Represents a loaded shared library."""
    name: str
    base_addr: int
    symbols: Dict[str, Symbol]
    relocations: List[Relocation]
    dependencies: List[str]


# ════════════════════════════════════════════════════════════════════════════════
# NEURAL SYMBOL RESOLVER
# ════════════════════════════════════════════════════════════════════════════════

class NeuralSymbolResolver(nn.Module):
    """
    Neural network for symbol name resolution using attention.

    Architecture:
    1. Character-level embedding of symbol names
    2. Positional encoding for name structure
    3. Transformer encoder to create symbol embeddings
    4. Cross-attention between query (undefined) and key (defined) symbols
    5. Confidence scoring for match quality

    This learns patterns like:
    - '_' prefix conventions
    - Version suffixes (@@GLIBC_2.17)
    - Common library naming patterns
    """

    def __init__(
        self,
        d_model: int = 128,
        max_symbol_len: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        max_symbols: int = 4096,
        device=None
    ):
        super().__init__()
        self.d_model = d_model
        self.max_symbol_len = max_symbol_len
        self.max_symbols = max_symbols
        self.device = device or torch.device('cpu')

        # Character embedding (ASCII range + special tokens)
        self.char_embed = nn.Embedding(256 + 2, d_model // 2)  # +2 for PAD and UNK
        self.PAD_TOKEN = 256
        self.UNK_TOKEN = 257

        # Learned positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_symbol_len, d_model // 2) * 0.02)

        # Project to full d_model
        self.input_proj = nn.Linear(d_model, d_model)

        # Transformer encoder for symbol names
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.name_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cross-attention: query (undefined refs) attends to keys (definitions)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # Output heads
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        # CLS token for pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.to(self.device)

    def encode_symbol_name(self, name: str) -> torch.Tensor:
        """Encode a single symbol name to embedding."""
        return self.encode_symbol_names([name])[0]

    def encode_symbol_names(self, names: List[str]) -> torch.Tensor:
        """
        Encode a batch of symbol names to embeddings.

        Args:
            names: List of symbol name strings

        Returns:
            Tensor of shape [batch, d_model] containing symbol embeddings
        """
        batch_size = len(names)

        # Convert names to character indices
        char_indices = torch.full(
            (batch_size, self.max_symbol_len),
            self.PAD_TOKEN,
            dtype=torch.long,
            device=self.device
        )

        for i, name in enumerate(names):
            for j, char in enumerate(name[:self.max_symbol_len]):
                char_indices[i, j] = ord(char) if ord(char) < 256 else self.UNK_TOKEN

        # Embed characters
        char_emb = self.char_embed(char_indices)  # [batch, seq, d_model//2]

        # Add positional embedding
        pos_emb = self.pos_embed[:, :self.max_symbol_len, :]  # [1, seq, d_model//2]
        combined = torch.cat([char_emb, pos_emb.expand(batch_size, -1, -1)], dim=-1)

        # Project to d_model
        x = self.input_proj(combined)  # [batch, seq, d_model]

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, 1+seq, d_model]

        # Create attention mask for padding
        pad_mask = char_indices == self.PAD_TOKEN
        # Prepend False for CLS token
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device)
        pad_mask = torch.cat([cls_mask, pad_mask], dim=1)

        # Encode with transformer
        encoded = self.name_encoder(x, src_key_padding_mask=pad_mask)

        # Return CLS token embedding as symbol representation
        return encoded[:, 0, :]  # [batch, d_model]

    def resolve(
        self,
        query_names: List[str],
        key_names: List[str],
        key_addresses: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Resolve query symbols against key symbols.

        Args:
            query_names: List of undefined symbol names to resolve
            key_names: List of defined symbol names (from libraries)
            key_addresses: Tensor of addresses for key symbols

        Returns:
            resolved_addresses: Tensor of resolved addresses for each query
            confidences: Tensor of confidence scores for each resolution
        """
        if not query_names or not key_names:
            return torch.zeros(len(query_names), device=self.device), \
                   torch.zeros(len(query_names), device=self.device)

        # Encode query and key symbol names
        query_emb = self.encode_symbol_names(query_names)  # [num_queries, d_model]
        key_emb = self.encode_symbol_names(key_names)      # [num_keys, d_model]

        # Cross-attention: queries attend to keys
        # Reshape for attention: [batch=1, seq, d_model]
        query_emb = query_emb.unsqueeze(0)
        key_emb = key_emb.unsqueeze(0)

        attn_output, attn_weights = self.cross_attn(
            query_emb, key_emb, key_emb
        )
        # attn_weights: [1, num_queries, num_keys]

        attn_weights = attn_weights.squeeze(0)  # [num_queries, num_keys]

        # Get best match for each query
        best_match_idx = attn_weights.argmax(dim=-1)  # [num_queries]
        resolved_addresses = key_addresses[best_match_idx]

        # Compute confidence from attention output
        confidences = self.confidence_head(attn_output.squeeze(0)).squeeze(-1)

        return resolved_addresses, confidences

    def forward(
        self,
        query_names: List[str],
        key_names: List[str],
        key_addresses: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass - alias for resolve()."""
        return self.resolve(query_names, key_names, key_addresses)


# ════════════════════════════════════════════════════════════════════════════════
# SYMBOL TABLE EMBEDDING CACHE
# ════════════════════════════════════════════════════════════════════════════════

class SymbolTableEmbedding:
    """
    Pre-computed embeddings for loaded library symbol tables.

    Caches symbol embeddings for fast repeated lookups during relocation.
    """

    def __init__(self, resolver: NeuralSymbolResolver):
        self.resolver = resolver
        self.device = resolver.device

        # Cached embeddings by library
        self._library_embeddings: Dict[str, torch.Tensor] = {}
        self._library_names: Dict[str, List[str]] = {}
        self._library_addresses: Dict[str, torch.Tensor] = {}

        # Global symbol table (merged from all libraries)
        self._global_names: List[str] = []
        self._global_addresses: torch.Tensor = None
        self._global_embeddings: torch.Tensor = None

    def load_library_symbols(self, library: LoadedLibrary):
        """
        Pre-compute embeddings for all symbols in a library.
        """
        if not library.symbols:
            return

        names = []
        addresses = []
        for sym in library.symbols.values():
            if sym.binding > 0:  # Global or weak
                names.append(sym.name)
                addresses.append(sym.address + library.base_addr)

        if not names:
            return

        # Compute embeddings
        with torch.no_grad():
            embeddings = self.resolver.encode_symbol_names(names)

        # Cache
        self._library_embeddings[library.name] = embeddings
        self._library_names[library.name] = names
        self._library_addresses[library.name] = torch.tensor(
            addresses, dtype=torch.int64, device=self.device
        )

        # Invalidate global cache
        self._global_embeddings = None

    def _build_global_table(self):
        """Build merged global symbol table from all libraries."""
        if self._global_embeddings is not None:
            return

        all_names = []
        all_addresses = []
        all_embeddings = []

        for lib_name in self._library_names:
            all_names.extend(self._library_names[lib_name])
            all_addresses.append(self._library_addresses[lib_name])
            all_embeddings.append(self._library_embeddings[lib_name])

        self._global_names = all_names
        if all_addresses:
            self._global_addresses = torch.cat(all_addresses)
            self._global_embeddings = torch.cat(all_embeddings)
        else:
            self._global_addresses = torch.tensor([], dtype=torch.int64, device=self.device)
            self._global_embeddings = torch.zeros(0, self.resolver.d_model, device=self.device)

    def resolve_symbols(
        self,
        undefined_names: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Resolve undefined symbols against all loaded libraries.

        Returns:
            addresses: Resolved addresses
            confidences: Match confidences
            matched_names: Names of matched definitions
        """
        self._build_global_table()

        if not undefined_names or len(self._global_names) == 0:
            return (
                torch.zeros(len(undefined_names), dtype=torch.int64, device=self.device),
                torch.zeros(len(undefined_names), device=self.device),
                [""] * len(undefined_names)
            )

        # Encode query symbols
        with torch.no_grad():
            query_emb = self.resolver.encode_symbol_names(undefined_names)

        # Compute similarity against global table
        # Using cosine similarity for matching
        query_norm = F.normalize(query_emb, dim=-1)
        key_norm = F.normalize(self._global_embeddings, dim=-1)
        similarity = torch.matmul(query_norm, key_norm.T)  # [num_queries, num_keys]

        # Get best matches
        best_scores, best_indices = similarity.max(dim=-1)
        addresses = self._global_addresses[best_indices]
        matched_names = [self._global_names[i] for i in best_indices.tolist()]

        # Convert similarity to confidence
        confidences = (best_scores + 1) / 2  # Map [-1,1] to [0,1]

        return addresses, confidences, matched_names


# ════════════════════════════════════════════════════════════════════════════════
# NEURAL RELOCATION ENGINE (MoE)
# ════════════════════════════════════════════════════════════════════════════════

class NeuralRelocationEngine(nn.Module):
    """
    Mixture-of-Experts network for relocation type handling.

    Each expert specializes in a relocation type:
    - RELATIVE expert: *offset = base + addend
    - ABS64 expert: *offset = symbol + addend
    - GLOB_DAT expert: GOT entries
    - JUMP_SLOT expert: PLT entries

    The router learns which expert to use based on relocation metadata.
    """

    def __init__(self, d_model: int = 64, device=None):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.d_model = d_model

        # Number of relocation experts
        self.num_experts = 5

        # Relocation type to expert mapping
        self.type_to_expert = {
            ARM64Reloc.R_AARCH64_NONE: 0,
            ARM64Reloc.R_AARCH64_RELATIVE: 1,
            ARM64Reloc.R_AARCH64_ABS64: 2,
            ARM64Reloc.R_AARCH64_GLOB_DAT: 3,
            ARM64Reloc.R_AARCH64_JUMP_SLOT: 4,
        }

        # Router network: determines expert weights from relocation context
        self.router = nn.Sequential(
            nn.Linear(32, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.num_experts),
            nn.Softmax(dim=-1)
        )

        # Expert networks - each learns a relocation formula
        # Input: [offset, addend, symbol_addr, base_addr] encoded
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(4, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1)
            )
            for _ in range(self.num_experts)
        ])

        self.to(self.device)

    def encode_relocation(self, reloc: Relocation, base_addr: int, symbol_addr: int) -> torch.Tensor:
        """Encode relocation context for routing."""
        # One-hot encode relocation type (up to 32 types)
        type_onehot = torch.zeros(32, device=self.device)
        type_idx = min(reloc.type, 31)
        type_onehot[type_idx] = 1.0
        return type_onehot

    def apply_relocation_batch(
        self,
        relocations: List[Relocation],
        base_addr: int,
        symbol_addresses: torch.Tensor,
        memory: torch.Tensor
    ) -> int:
        """
        Apply a batch of relocations using neural routing.

        This is the hybrid approach: use learned routing but apply
        exact relocation formulas for correctness.

        Returns:
            Number of relocations applied
        """
        applied = 0

        for i, reloc in enumerate(relocations):
            symbol_addr = int(symbol_addresses[i].item()) if i < len(symbol_addresses) else 0

            # Get expert routing
            context = self.encode_relocation(reloc, base_addr, symbol_addr)
            with torch.no_grad():
                expert_weights = self.router(context.unsqueeze(0)).squeeze(0)
            expert_idx = expert_weights.argmax().item()

            # Apply exact relocation formula based on type
            result = self._apply_single_relocation(
                reloc, base_addr, symbol_addr, memory
            )

            if result:
                applied += 1

        return applied

    def _apply_single_relocation(
        self,
        reloc: Relocation,
        base_addr: int,
        symbol_addr: int,
        memory: torch.Tensor
    ) -> bool:
        """Apply a single relocation with exact formula."""
        offset = reloc.offset
        addend = reloc.addend
        rtype = reloc.type

        # Validate offset
        if offset < 0 or offset + 8 > len(memory):
            return False

        # Compute value based on relocation type
        if rtype == ARM64Reloc.R_AARCH64_RELATIVE:
            # S + A where S = base_addr
            value = base_addr + addend

        elif rtype == ARM64Reloc.R_AARCH64_ABS64:
            # S + A where S = symbol address
            value = symbol_addr + addend

        elif rtype == ARM64Reloc.R_AARCH64_GLOB_DAT:
            # S (symbol address into GOT)
            value = symbol_addr

        elif rtype == ARM64Reloc.R_AARCH64_JUMP_SLOT:
            # S (symbol address into PLT GOT)
            value = symbol_addr

        elif rtype == ARM64Reloc.R_AARCH64_NONE:
            return True  # No-op

        else:
            # Unknown relocation type
            return False

        # Write 64-bit value to memory (little-endian)
        value_bytes = struct.pack('<Q', value & 0xFFFFFFFFFFFFFFFF)
        for j, b in enumerate(value_bytes):
            memory[offset + j] = b

        return True


# ════════════════════════════════════════════════════════════════════════════════
# NEURAL DYNAMIC LINKER ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════════

class NeuralDynamicLinker:
    """
    Orchestrates the full neural dynamic linking process.

    Steps:
    1. Parse executable and find dependencies (DT_NEEDED)
    2. Load libraries in dependency order
    3. Neural symbol resolution for undefined references
    4. Apply relocations using MoE engine
    5. Return entry point

    This is the main interface for loading dynamically linked binaries.
    """

    def __init__(self, cpu, kernel, device=None):
        self.cpu = cpu
        self.kernel = kernel
        self.device = device or cpu.device

        # Neural components
        self.symbol_resolver = NeuralSymbolResolver(device=self.device)
        self.symbol_cache = SymbolTableEmbedding(self.symbol_resolver)
        self.reloc_engine = NeuralRelocationEngine(device=self.device)

        # Loaded libraries
        self.libraries: Dict[str, LoadedLibrary] = {}
        self.load_order: List[str] = []

        # Library search paths
        self.library_paths = [
            '/lib',
            '/lib64',
            '/usr/lib',
            '/usr/lib64',
            '/lib/aarch64-linux-gnu',
            '/usr/lib/aarch64-linux-gnu',
        ]

        # Stats
        self.stats = {
            'symbols_resolved': 0,
            'relocations_applied': 0,
            'libraries_loaded': 0,
            'resolution_confidence_avg': 0.0
        }

    def load_executable(self, elf_data: bytes, exe_path: str = "/usr/bin/program") -> int:
        """
        Load a dynamically linked executable.

        Returns:
            Entry point address, or -1 on failure
        """
        # Store executable path for /proc/self/exe
        self.kernel._current_exe_path = exe_path

        # Parse ELF header
        entry_point, interp, dependencies = self._parse_elf(elf_data)

        if interp:
            # Has interpreter (PT_INTERP) - load it first
            interp_data = self._find_library(interp)
            if interp_data:
                interp_entry = self._load_library(interp, interp_data, is_interp=True)
                if interp_entry > 0:
                    # Run interpreter which will handle the rest
                    return interp_entry

        # No interpreter or failed to load - try direct loading
        # Load dependencies
        for dep in dependencies:
            self._load_dependency(dep)

        # Resolve symbols and apply relocations
        self._resolve_and_relocate()

        return entry_point

    def _parse_elf(self, data: bytes) -> Tuple[int, Optional[str], List[str]]:
        """Parse ELF to get entry, interpreter, and dependencies."""
        if len(data) < 64 or data[:4] != b'\x7fELF':
            return -1, None, []

        # Parse ELF header
        e_entry = struct.unpack('<Q', data[24:32])[0]
        e_phoff = struct.unpack('<Q', data[32:40])[0]
        e_phentsize = struct.unpack('<H', data[54:56])[0]
        e_phnum = struct.unpack('<H', data[56:58])[0]

        interp = None
        dependencies = []

        # Parse program headers
        for i in range(e_phnum):
            ph_offset = e_phoff + i * e_phentsize
            if ph_offset + 56 > len(data):
                break

            p_type = struct.unpack('<I', data[ph_offset:ph_offset+4])[0]
            p_offset = struct.unpack('<Q', data[ph_offset+8:ph_offset+16])[0]
            p_filesz = struct.unpack('<Q', data[ph_offset+32:ph_offset+40])[0]

            # PT_INTERP
            if p_type == 3:
                interp_end = data.find(b'\x00', p_offset)
                if interp_end > p_offset:
                    interp = data[p_offset:interp_end].decode('utf-8', errors='ignore')

            # PT_DYNAMIC
            if p_type == 2:
                dependencies = self._parse_dynamic(data, p_offset, p_filesz)

        return e_entry, interp, dependencies

    def _parse_dynamic(self, data: bytes, offset: int, size: int) -> List[str]:
        """Parse DT_NEEDED entries from PT_DYNAMIC."""
        dependencies = []
        strtab_offset = 0

        # First pass: find STRTAB
        pos = offset
        while pos < offset + size:
            if pos + 16 > len(data):
                break
            d_tag = struct.unpack('<Q', data[pos:pos+8])[0]
            d_val = struct.unpack('<Q', data[pos+8:pos+16])[0]

            if d_tag == 5:  # DT_STRTAB
                strtab_offset = d_val
            if d_tag == 0:  # DT_NULL
                break

            pos += 16

        # Second pass: collect DT_NEEDED
        if strtab_offset > 0:
            pos = offset
            while pos < offset + size:
                if pos + 16 > len(data):
                    break
                d_tag = struct.unpack('<Q', data[pos:pos+8])[0]
                d_val = struct.unpack('<Q', data[pos+8:pos+16])[0]

                if d_tag == 1:  # DT_NEEDED
                    str_offset = strtab_offset + d_val
                    if str_offset < len(data):
                        str_end = data.find(b'\x00', str_offset)
                        if str_end > str_offset:
                            lib_name = data[str_offset:str_end].decode('utf-8', errors='ignore')
                            dependencies.append(lib_name)
                if d_tag == 0:
                    break

                pos += 16

        return dependencies

    def _find_library(self, name: str) -> Optional[bytes]:
        """Find and load library from sysroot."""
        import os

        # Get sysroot
        sysroot = os.environ.get('NEURAL_SYSROOT', '')
        if not sysroot:
            sysroot = getattr(self.kernel, 'sysroot', '')

        if not sysroot:
            return None

        # Search paths
        for search_path in self.library_paths:
            full_path = os.path.join(sysroot, search_path.lstrip('/'), name)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'rb') as f:
                        return f.read()
                except Exception:
                    pass

        return None

    def _load_library(self, name: str, data: bytes, is_interp: bool = False) -> int:
        """Load a shared library into memory."""
        # TODO: Full library loading implementation
        # For now, return -1 to indicate not yet implemented
        self.stats['libraries_loaded'] += 1
        return -1

    def _load_dependency(self, name: str):
        """Load a library dependency."""
        if name in self.libraries:
            return

        data = self._find_library(name)
        if data:
            self._load_library(name, data)

    def _resolve_and_relocate(self):
        """Resolve all undefined symbols and apply relocations."""
        # Collect all undefined symbols from all loaded objects
        undefined = []
        for lib in self.libraries.values():
            for reloc in lib.relocations:
                if reloc.symbol_name:
                    undefined.append(reloc.symbol_name)

        if not undefined:
            return

        # Resolve using neural symbol resolver
        addresses, confidences, matched = self.symbol_cache.resolve_symbols(undefined)

        self.stats['symbols_resolved'] = len(undefined)
        self.stats['resolution_confidence_avg'] = float(confidences.mean()) if len(confidences) > 0 else 0.0

        # Apply relocations
        for lib in self.libraries.values():
            applied = self.reloc_engine.apply_relocation_batch(
                lib.relocations,
                lib.base_addr,
                addresses,
                self.cpu.memory
            )
            self.stats['relocations_applied'] += applied

    def get_stats(self) -> dict:
        """Get linker statistics."""
        return self.stats.copy()


# ════════════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ════════════════════════════════════════════════════════════════════════════════

def generate_symbol_training_data(
    num_samples: int = 10000
) -> Tuple[List[str], List[str], torch.Tensor]:
    """
    Generate synthetic training data for symbol resolution.

    Creates pairs of (undefined_name, defined_name) where they should match.
    """
    import random
    import string

    # Common library function prefixes
    prefixes = ['', '_', '__', '___']
    suffixes = ['', '_impl', '_internal', '_v2', '_default']
    version_suffixes = ['', '@@GLIBC_2.17', '@@GLIBC_2.4', '@GLIBC_2.0']

    # Common function name patterns
    base_names = [
        'malloc', 'free', 'printf', 'strlen', 'strcpy', 'memcpy', 'memset',
        'open', 'close', 'read', 'write', 'exit', 'abort', 'signal',
        'pthread_create', 'pthread_mutex_lock', 'dlopen', 'dlsym',
    ]

    queries = []
    keys = []
    labels = []

    for _ in range(num_samples):
        # Pick a base name
        base = random.choice(base_names)

        # Create variants
        query_prefix = random.choice(prefixes)
        key_prefix = random.choice(prefixes)
        version = random.choice(version_suffixes)

        query = query_prefix + base
        key = key_prefix + base + version

        queries.append(query)
        keys.append(key)
        labels.append(1)  # Positive pair

        # Add some negative pairs
        if random.random() < 0.5:
            wrong_base = random.choice(base_names)
            while wrong_base == base:
                wrong_base = random.choice(base_names)
            queries.append(query_prefix + base)
            keys.append(key_prefix + wrong_base + version)
            labels.append(0)  # Negative pair

    return queries, keys, torch.tensor(labels, dtype=torch.float32)


if __name__ == "__main__":
    print("=" * 70)
    print("  Neural Dynamic Linker - Component Test")
    print("=" * 70)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Test Symbol Resolver
    print("\n1. Testing NeuralSymbolResolver...")
    resolver = NeuralSymbolResolver(device=device)
    total_params = sum(p.numel() for p in resolver.parameters())
    print(f"   Parameters: {total_params:,}")

    # Test encoding
    test_names = ['malloc', 'free', 'printf', '__libc_start_main']
    embeddings = resolver.encode_symbol_names(test_names)
    print(f"   Encoded {len(test_names)} symbols: shape={embeddings.shape}")

    # Test resolution
    query_names = ['malloc', 'printf']
    key_names = ['malloc', 'free', 'printf', 'strlen']
    key_addrs = torch.tensor([0x1000, 0x2000, 0x3000, 0x4000], device=device)

    addrs, confs = resolver.resolve(query_names, key_names, key_addrs)
    print(f"   Resolution test:")
    for i, q in enumerate(query_names):
        print(f"     {q} -> addr=0x{int(addrs[i]):X}, conf={float(confs[i]):.3f}")

    # Test Relocation Engine
    print("\n2. Testing NeuralRelocationEngine...")
    reloc_engine = NeuralRelocationEngine(device=device)
    reloc_params = sum(p.numel() for p in reloc_engine.parameters())
    print(f"   Parameters: {reloc_params:,}")

    print("\n✅ Neural Dynamic Linker components initialized successfully!")
    print("=" * 70)
