#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    NEURAL SEMANTIC SYSTEM                                     ║
║                                                                              ║
║  A semantic understanding layer for NeuralOS that comprehends:               ║
║  - Program intent and behavior patterns                                      ║
║  - User goals and task context                                               ║
║  - System state and resource semantics                                       ║
║  - Natural language ↔ execution mapping                                      ║
║                                                                              ║
║  This goes beyond execution to UNDERSTANDING what programs mean.             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum, auto
from collections import defaultdict
import math
import time

# ════════════════════════════════════════════════════════════════════════════════
# DEVICE CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ════════════════════════════════════════════════════════════════════════════════
# SEMANTIC CONCEPTS - The "vocabulary" of system understanding
# ════════════════════════════════════════════════════════════════════════════════

class SemanticConcept(Enum):
    """Core semantic concepts the system understands."""
    # File Operations
    FILE_READ = auto()
    FILE_WRITE = auto()
    FILE_CREATE = auto()
    FILE_DELETE = auto()
    FILE_LIST = auto()

    # Process Operations
    PROCESS_START = auto()
    PROCESS_STOP = auto()
    PROCESS_COMMUNICATE = auto()

    # System Information
    SYSTEM_QUERY = auto()
    SYSTEM_CONFIGURE = auto()
    SYSTEM_MONITOR = auto()

    # User Interaction
    USER_INPUT = auto()
    USER_OUTPUT = auto()
    USER_NOTIFY = auto()

    # Data Operations
    DATA_TRANSFORM = auto()
    DATA_FILTER = auto()
    DATA_AGGREGATE = auto()
    DATA_SEARCH = auto()

    # Network Operations
    NETWORK_CONNECT = auto()
    NETWORK_SEND = auto()
    NETWORK_RECEIVE = auto()

    # Security Operations
    SECURITY_AUTH = auto()
    SECURITY_ENCRYPT = auto()
    SECURITY_VALIDATE = auto()

    # Resource Management
    RESOURCE_ALLOCATE = auto()
    RESOURCE_FREE = auto()
    RESOURCE_SHARE = auto()


class IntentType(Enum):
    """High-level user intents."""
    EXPLORE = auto()      # User wants to understand/explore
    CREATE = auto()       # User wants to make something new
    MODIFY = auto()       # User wants to change something
    DELETE = auto()       # User wants to remove something
    QUERY = auto()        # User wants information
    AUTOMATE = auto()     # User wants repeated/scheduled action
    DEBUG = auto()        # User is troubleshooting
    OPTIMIZE = auto()     # User wants better performance


@dataclass
class SemanticFrame:
    """A semantic frame representing understood meaning."""
    concepts: List[SemanticConcept]
    intent: IntentType
    entities: Dict[str, str]  # Named entities extracted
    confidence: float
    context: Dict[str, any] = field(default_factory=dict)

    def __repr__(self):
        return f"SemanticFrame(intent={self.intent.name}, concepts={[c.name for c in self.concepts]}, conf={self.confidence:.2f})"


# ════════════════════════════════════════════════════════════════════════════════
# SEMANTIC EMBEDDING NETWORK - Learns distributed representations of meaning
# ════════════════════════════════════════════════════════════════════════════════

class SemanticEmbeddingNetwork(nn.Module):
    """
    Neural network that learns semantic embeddings for:
    - Commands and programs
    - Syscalls and operations
    - User patterns and intents

    Uses attention to relate different semantic concepts.
    """

    def __init__(self, vocab_size: int = 1024, embed_dim: int = 128, n_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim

        # Token embeddings for command vocabulary
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Concept embeddings (one per SemanticConcept)
        self.concept_embed = nn.Embedding(len(SemanticConcept), embed_dim)

        # Intent embeddings (one per IntentType)
        self.intent_embed = nn.Embedding(len(IntentType), embed_dim)

        # Positional encoding
        self.pos_embed = nn.Embedding(256, embed_dim)

        # Multi-head self-attention for relating concepts
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

        # Feed-forward layers
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Output projections
        self.concept_classifier = nn.Linear(embed_dim, len(SemanticConcept))
        self.intent_classifier = nn.Linear(embed_dim, len(IntentType))
        self.confidence_head = nn.Linear(embed_dim, 1)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process token sequence and extract semantic meaning.

        Returns:
            concept_logits: Probability of each semantic concept
            intent_logits: Probability of each intent type
            confidence: Confidence in the understanding
        """
        B, T = tokens.shape

        # Embed tokens with positions
        positions = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(tokens) + self.pos_embed(positions)

        # Self-attention to relate tokens
        attended, _ = self.attention(x, x, x)
        x = x + attended

        # Feed-forward processing
        x = self.ff(x)

        # Pool across sequence (mean pooling)
        pooled = x.mean(dim=1)

        # Classify concepts and intent
        concept_logits = self.concept_classifier(pooled)
        intent_logits = self.intent_classifier(pooled)
        confidence = torch.sigmoid(self.confidence_head(pooled))

        return concept_logits, intent_logits, confidence


# ════════════════════════════════════════════════════════════════════════════════
# SYSCALL SEMANTIC MAPPER - Maps low-level syscalls to high-level meaning
# ════════════════════════════════════════════════════════════════════════════════

class SyscallSemanticMapper:
    """
    Maps syscall numbers and patterns to semantic concepts.
    Learns relationships between low-level operations and high-level meaning.
    """

    # Static mapping of syscalls to semantic concepts
    SYSCALL_SEMANTICS = {
        # File operations
        56: [SemanticConcept.FILE_CREATE],      # openat
        57: [SemanticConcept.FILE_DELETE],      # close (cleanup)
        63: [SemanticConcept.FILE_READ],        # read
        64: [SemanticConcept.FILE_WRITE, SemanticConcept.USER_OUTPUT],  # write

        # Directory operations
        17: [SemanticConcept.FILE_LIST],        # getcwd
        34: [SemanticConcept.FILE_LIST],        # mkdirat
        35: [SemanticConcept.FILE_DELETE],      # unlinkat

        # Process operations
        93: [SemanticConcept.PROCESS_STOP],     # exit
        94: [SemanticConcept.PROCESS_STOP],     # exit_group
        220: [SemanticConcept.PROCESS_START],   # clone
        221: [SemanticConcept.PROCESS_START],   # execve

        # System info
        160: [SemanticConcept.SYSTEM_QUERY],    # uname
        169: [SemanticConcept.SYSTEM_QUERY],    # gettimeofday
        113: [SemanticConcept.SYSTEM_QUERY],    # clock_gettime

        # Memory operations
        222: [SemanticConcept.RESOURCE_ALLOCATE],  # mmap
        215: [SemanticConcept.RESOURCE_FREE],      # munmap
        226: [SemanticConcept.RESOURCE_ALLOCATE],  # mprotect

        # User/group
        174: [SemanticConcept.SECURITY_AUTH],   # getuid
        175: [SemanticConcept.SECURITY_AUTH],   # geteuid
        176: [SemanticConcept.SECURITY_AUTH],   # getgid

        # Signals
        135: [SemanticConcept.PROCESS_COMMUNICATE],  # sigaction
        139: [SemanticConcept.PROCESS_COMMUNICATE],  # sigreturn

        # I/O
        78: [SemanticConcept.FILE_READ],        # readlinkat
        79: [SemanticConcept.FILE_LIST],        # newfstatat
        80: [SemanticConcept.FILE_LIST],        # fstat
    }

    def __init__(self):
        self.syscall_patterns = defaultdict(list)
        self.pattern_meanings = {}

    def map_syscall(self, syscall_num: int) -> List[SemanticConcept]:
        """Map a single syscall to its semantic concepts."""
        return self.SYSCALL_SEMANTICS.get(syscall_num, [])

    def analyze_syscall_sequence(self, syscalls: List[int]) -> SemanticFrame:
        """
        Analyze a sequence of syscalls to understand overall meaning.

        Patterns like [openat, read, close] → FILE_READ intent
        Patterns like [openat, write, close] → FILE_WRITE intent
        """
        concepts = set()

        # Gather all concepts from individual syscalls
        for sc in syscalls:
            concepts.update(self.map_syscall(sc))

        # Pattern detection for intent
        intent = self._detect_intent_from_pattern(syscalls, concepts)

        return SemanticFrame(
            concepts=list(concepts),
            intent=intent,
            entities={},
            confidence=0.8 if concepts else 0.3
        )

    def _detect_intent_from_pattern(self, syscalls: List[int], concepts: Set[SemanticConcept]) -> IntentType:
        """Detect high-level intent from syscall patterns and concepts."""

        # File reading pattern
        if SemanticConcept.FILE_READ in concepts and SemanticConcept.USER_OUTPUT in concepts:
            return IntentType.QUERY

        # File creation pattern
        if SemanticConcept.FILE_CREATE in concepts and SemanticConcept.FILE_WRITE in concepts:
            return IntentType.CREATE

        # System query pattern
        if SemanticConcept.SYSTEM_QUERY in concepts:
            return IntentType.QUERY

        # Process management
        if SemanticConcept.PROCESS_START in concepts:
            return IntentType.AUTOMATE

        # Default to exploration
        return IntentType.EXPLORE


# ════════════════════════════════════════════════════════════════════════════════
# COMMAND SEMANTIC ANALYZER - Understands command names and arguments
# ════════════════════════════════════════════════════════════════════════════════

class CommandSemanticAnalyzer:
    """
    Analyzes command names and arguments to understand intent.
    Uses pattern matching and learned associations.
    """

    COMMAND_SEMANTICS = {
        # File commands
        'cat': (IntentType.QUERY, [SemanticConcept.FILE_READ, SemanticConcept.USER_OUTPUT]),
        'ls': (IntentType.EXPLORE, [SemanticConcept.FILE_LIST]),
        'echo': (IntentType.CREATE, [SemanticConcept.USER_OUTPUT]),
        'touch': (IntentType.CREATE, [SemanticConcept.FILE_CREATE]),
        'rm': (IntentType.DELETE, [SemanticConcept.FILE_DELETE]),
        'cp': (IntentType.CREATE, [SemanticConcept.FILE_READ, SemanticConcept.FILE_CREATE]),
        'mv': (IntentType.MODIFY, [SemanticConcept.FILE_READ, SemanticConcept.FILE_DELETE, SemanticConcept.FILE_CREATE]),
        'mkdir': (IntentType.CREATE, [SemanticConcept.FILE_CREATE]),

        # System commands
        'uname': (IntentType.QUERY, [SemanticConcept.SYSTEM_QUERY]),
        'whoami': (IntentType.QUERY, [SemanticConcept.SECURITY_AUTH]),
        'hostname': (IntentType.QUERY, [SemanticConcept.SYSTEM_QUERY]),
        'uptime': (IntentType.QUERY, [SemanticConcept.SYSTEM_MONITOR]),
        'date': (IntentType.QUERY, [SemanticConcept.SYSTEM_QUERY]),
        'pwd': (IntentType.QUERY, [SemanticConcept.FILE_LIST]),

        # Process commands
        'ps': (IntentType.QUERY, [SemanticConcept.SYSTEM_MONITOR]),
        'top': (IntentType.QUERY, [SemanticConcept.SYSTEM_MONITOR]),
        'kill': (IntentType.DELETE, [SemanticConcept.PROCESS_STOP]),

        # Data commands
        'grep': (IntentType.QUERY, [SemanticConcept.DATA_SEARCH, SemanticConcept.DATA_FILTER]),
        'find': (IntentType.QUERY, [SemanticConcept.DATA_SEARCH]),
        'sort': (IntentType.MODIFY, [SemanticConcept.DATA_TRANSFORM]),
        'head': (IntentType.QUERY, [SemanticConcept.FILE_READ, SemanticConcept.DATA_FILTER]),
        'tail': (IntentType.QUERY, [SemanticConcept.FILE_READ, SemanticConcept.DATA_FILTER]),
        'wc': (IntentType.QUERY, [SemanticConcept.DATA_AGGREGATE]),
        'seq': (IntentType.CREATE, [SemanticConcept.DATA_TRANSFORM, SemanticConcept.USER_OUTPUT]),

        # Display commands
        'banner': (IntentType.CREATE, [SemanticConcept.USER_OUTPUT]),
        'neofetch': (IntentType.QUERY, [SemanticConcept.SYSTEM_QUERY, SemanticConcept.USER_OUTPUT]),

        # Special
        'true': (IntentType.AUTOMATE, []),
        'false': (IntentType.AUTOMATE, []),
        'yes': (IntentType.AUTOMATE, [SemanticConcept.USER_OUTPUT]),
    }

    def __init__(self):
        self.learned_commands = {}
        self.command_history = []

    def analyze_command(self, command: str, args: List[str] = None) -> SemanticFrame:
        """Analyze a command to understand its semantic meaning."""

        args = args or []
        base_cmd = command.split('/')[-1]  # Handle paths

        if base_cmd in self.COMMAND_SEMANTICS:
            intent, concepts = self.COMMAND_SEMANTICS[base_cmd]

            # Extract entities from arguments
            entities = self._extract_entities(base_cmd, args)

            return SemanticFrame(
                concepts=concepts,
                intent=intent,
                entities=entities,
                confidence=0.9,
                context={'command': base_cmd, 'args': args}
            )

        # Unknown command - try to infer from name
        return self._infer_unknown_command(base_cmd, args)

    def _extract_entities(self, command: str, args: List[str]) -> Dict[str, str]:
        """Extract named entities from command arguments."""
        entities = {}

        for i, arg in enumerate(args):
            if arg.startswith('-'):
                continue  # Skip flags

            # Detect file paths
            if '/' in arg or '.' in arg:
                entities[f'path_{i}'] = arg
            elif arg.isdigit():
                entities[f'number_{i}'] = arg
            else:
                entities[f'arg_{i}'] = arg

        return entities

    def _infer_unknown_command(self, command: str, args: List[str]) -> SemanticFrame:
        """Try to infer meaning from unknown command name."""

        concepts = []
        intent = IntentType.EXPLORE

        # Name-based heuristics
        if 'read' in command or 'get' in command or 'show' in command:
            concepts.append(SemanticConcept.FILE_READ)
            intent = IntentType.QUERY
        elif 'write' in command or 'set' in command or 'put' in command:
            concepts.append(SemanticConcept.FILE_WRITE)
            intent = IntentType.MODIFY
        elif 'make' in command or 'create' in command or 'new' in command:
            concepts.append(SemanticConcept.FILE_CREATE)
            intent = IntentType.CREATE
        elif 'del' in command or 'rm' in command or 'remove' in command:
            concepts.append(SemanticConcept.FILE_DELETE)
            intent = IntentType.DELETE
        elif 'list' in command or 'dir' in command:
            concepts.append(SemanticConcept.FILE_LIST)
            intent = IntentType.EXPLORE

        return SemanticFrame(
            concepts=concepts,
            intent=intent,
            entities=self._extract_entities(command, args),
            confidence=0.5,  # Low confidence for inferred
            context={'command': command, 'args': args, 'inferred': True}
        )

    def learn_command(self, command: str, observed_behavior: List[SemanticConcept]):
        """Learn a new command's semantics from observed behavior."""
        self.learned_commands[command] = observed_behavior


# ════════════════════════════════════════════════════════════════════════════════
# SEMANTIC MEMORY - Long-term storage of learned meanings
# ════════════════════════════════════════════════════════════════════════════════

class SemanticMemory:
    """
    Long-term semantic memory that stores and retrieves learned meanings.
    Uses neural embeddings for similarity-based retrieval.
    """

    def __init__(self, embed_dim: int = 128):
        self.embed_dim = embed_dim

        # Memory banks
        self.command_memories: Dict[str, torch.Tensor] = {}
        self.pattern_memories: List[Tuple[torch.Tensor, SemanticFrame]] = []
        self.context_memories: List[Dict] = []

        # Embedding network for computing similarities
        self.embed_network = nn.Sequential(
            nn.Linear(64, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        ).to(device)

    def store_command(self, command: str, frame: SemanticFrame):
        """Store semantic understanding of a command."""
        # Create embedding from concepts and intent
        concept_ids = torch.tensor([c.value for c in frame.concepts], dtype=torch.float32)
        if len(concept_ids) == 0:
            concept_ids = torch.zeros(1)

        # Pad/truncate to fixed size
        padded = torch.zeros(64)
        padded[:min(len(concept_ids), 64)] = concept_ids[:64]

        embedding = self.embed_network(padded.to(device))
        self.command_memories[command] = embedding.detach()

    def retrieve_similar(self, query_frame: SemanticFrame, k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve commands with similar semantics."""
        if not self.command_memories:
            return []

        # Create query embedding
        concept_ids = torch.tensor([c.value for c in query_frame.concepts], dtype=torch.float32)
        if len(concept_ids) == 0:
            concept_ids = torch.zeros(1)

        padded = torch.zeros(64)
        padded[:min(len(concept_ids), 64)] = concept_ids[:64]

        query_embed = self.embed_network(padded.to(device))

        # Compute similarities
        similarities = []
        for cmd, embed in self.command_memories.items():
            sim = F.cosine_similarity(query_embed, embed, dim=0).item()
            similarities.append((cmd, sim))

        # Return top-k
        similarities.sort(key=lambda x: -x[1])
        return similarities[:k]

    def store_context(self, context: Dict):
        """Store contextual information for later retrieval."""
        context['timestamp'] = time.time()
        self.context_memories.append(context)

        # Keep last 1000 contexts
        if len(self.context_memories) > 1000:
            self.context_memories = self.context_memories[-1000:]


# ════════════════════════════════════════════════════════════════════════════════
# NATURAL LANGUAGE INTERFACE - Map natural language to system operations
# ════════════════════════════════════════════════════════════════════════════════

class NaturalLanguageInterface:
    """
    Maps natural language queries to system operations.
    Enables semantic interaction with the OS.
    """

    # Simple keyword-based mapping (could be replaced with actual NLP)
    INTENT_KEYWORDS = {
        IntentType.EXPLORE: ['show', 'list', 'what', 'where', 'display', 'view'],
        IntentType.CREATE: ['make', 'create', 'new', 'write', 'add', 'generate'],
        IntentType.MODIFY: ['change', 'edit', 'update', 'modify', 'rename'],
        IntentType.DELETE: ['delete', 'remove', 'rm', 'erase', 'clear'],
        IntentType.QUERY: ['find', 'search', 'get', 'read', 'tell', 'who', 'when'],
        IntentType.AUTOMATE: ['run', 'execute', 'start', 'do', 'repeat'],
        IntentType.DEBUG: ['debug', 'trace', 'why', 'error', 'fix', 'problem'],
        IntentType.OPTIMIZE: ['faster', 'optimize', 'improve', 'speed', 'better'],
    }

    CONCEPT_KEYWORDS = {
        SemanticConcept.FILE_READ: ['file', 'read', 'open', 'load'],
        SemanticConcept.FILE_WRITE: ['save', 'write', 'store'],
        SemanticConcept.FILE_LIST: ['files', 'directory', 'folder', 'contents'],
        SemanticConcept.SYSTEM_QUERY: ['system', 'info', 'status', 'version'],
        SemanticConcept.PROCESS_START: ['process', 'program', 'run', 'app'],
        SemanticConcept.DATA_SEARCH: ['find', 'search', 'look', 'grep'],
    }

    COMMAND_MAPPING = {
        ('show', 'files'): 'ls',
        ('list', 'files'): 'ls',
        ('what', 'files'): 'ls',
        ('show', 'directory'): 'ls',
        ('who', 'am'): 'whoami',
        ('current', 'user'): 'whoami',
        ('system', 'info'): 'uname',
        ('show', 'system'): 'neofetch',
        ('current', 'directory'): 'pwd',
        ('where', 'am'): 'pwd',
        ('current', 'time'): 'date',
        ('what', 'time'): 'date',
        ('show', 'banner'): 'banner',
        ('count', 'to'): 'seq',
        ('uptime',): 'uptime',
        ('hostname',): 'hostname',
    }

    def __init__(self):
        self.command_analyzer = CommandSemanticAnalyzer()

    def parse_query(self, query: str) -> SemanticFrame:
        """Parse natural language query into semantic frame."""
        tokens = query.lower().split()

        # Detect intent
        intent = self._detect_intent(tokens)

        # Detect concepts
        concepts = self._detect_concepts(tokens)

        # Extract entities
        entities = self._extract_entities_from_nl(tokens)

        return SemanticFrame(
            concepts=concepts,
            intent=intent,
            entities=entities,
            confidence=0.7,
            context={'original_query': query, 'tokens': tokens}
        )

    def query_to_command(self, query: str) -> Optional[Tuple[str, List[str]]]:
        """Convert natural language query to a command."""
        tokens = query.lower().split()

        # Try direct mappings first
        for key_tuple, cmd in self.COMMAND_MAPPING.items():
            if all(k in tokens for k in key_tuple):
                return (cmd, [])

        # Try single-word command names
        for token in tokens:
            if token in CommandSemanticAnalyzer.COMMAND_SEMANTICS:
                return (token, [])

        return None

    def _detect_intent(self, tokens: List[str]) -> IntentType:
        """Detect intent from tokens."""
        for intent, keywords in self.INTENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in tokens:
                    return intent
        return IntentType.EXPLORE

    def _detect_concepts(self, tokens: List[str]) -> List[SemanticConcept]:
        """Detect concepts from tokens."""
        concepts = []
        for concept, keywords in self.CONCEPT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in tokens and concept not in concepts:
                    concepts.append(concept)
        return concepts

    def _extract_entities_from_nl(self, tokens: List[str]) -> Dict[str, str]:
        """Extract entities from natural language."""
        entities = {}

        for i, token in enumerate(tokens):
            # Detect numbers
            if token.isdigit():
                entities[f'number_{i}'] = token
            # Detect paths
            elif '/' in token or '.' in token:
                entities[f'path_{i}'] = token

        return entities


# ════════════════════════════════════════════════════════════════════════════════
# SEMANTIC REASONING ENGINE - Performs inference over semantic knowledge
# ════════════════════════════════════════════════════════════════════════════════

class SemanticReasoningEngine:
    """
    Performs semantic reasoning to:
    - Infer missing information
    - Predict consequences of actions
    - Suggest alternatives
    - Explain behavior
    """

    # Concept relationships (A implies B)
    CONCEPT_IMPLICATIONS = {
        SemanticConcept.FILE_WRITE: [SemanticConcept.RESOURCE_ALLOCATE],
        SemanticConcept.FILE_CREATE: [SemanticConcept.FILE_WRITE],
        SemanticConcept.PROCESS_START: [SemanticConcept.RESOURCE_ALLOCATE],
        SemanticConcept.NETWORK_CONNECT: [SemanticConcept.RESOURCE_ALLOCATE],
    }

    # Concept conflicts (A conflicts with B)
    CONCEPT_CONFLICTS = {
        SemanticConcept.FILE_READ: [SemanticConcept.FILE_DELETE],
        SemanticConcept.PROCESS_START: [SemanticConcept.PROCESS_STOP],
    }

    def __init__(self):
        self.inference_cache = {}

    def infer_implications(self, concepts: List[SemanticConcept]) -> List[SemanticConcept]:
        """Infer additional concepts implied by the given ones."""
        inferred = set(concepts)

        changed = True
        while changed:
            changed = False
            for concept in list(inferred):
                if concept in self.CONCEPT_IMPLICATIONS:
                    for implied in self.CONCEPT_IMPLICATIONS[concept]:
                        if implied not in inferred:
                            inferred.add(implied)
                            changed = True

        return list(inferred)

    def detect_conflicts(self, concepts: List[SemanticConcept]) -> List[Tuple[SemanticConcept, SemanticConcept]]:
        """Detect conflicting concepts."""
        conflicts = []

        for concept in concepts:
            if concept in self.CONCEPT_CONFLICTS:
                for conflicting in self.CONCEPT_CONFLICTS[concept]:
                    if conflicting in concepts:
                        conflicts.append((concept, conflicting))

        return conflicts

    def explain_behavior(self, frame: SemanticFrame) -> str:
        """Generate human-readable explanation of semantic behavior."""

        intent_explanations = {
            IntentType.EXPLORE: "exploring/discovering",
            IntentType.CREATE: "creating/generating",
            IntentType.MODIFY: "modifying/changing",
            IntentType.DELETE: "deleting/removing",
            IntentType.QUERY: "querying/requesting",
            IntentType.AUTOMATE: "automating/executing",
            IntentType.DEBUG: "debugging/troubleshooting",
            IntentType.OPTIMIZE: "optimizing/improving",
        }

        concept_explanations = {
            SemanticConcept.FILE_READ: "reading files",
            SemanticConcept.FILE_WRITE: "writing files",
            SemanticConcept.FILE_LIST: "listing directory contents",
            SemanticConcept.SYSTEM_QUERY: "querying system information",
            SemanticConcept.USER_OUTPUT: "displaying output to user",
            SemanticConcept.DATA_TRANSFORM: "transforming data",
            SemanticConcept.SECURITY_AUTH: "checking authentication",
        }

        parts = [f"Intent: {intent_explanations.get(frame.intent, 'unknown')}"]

        if frame.concepts:
            concept_strs = [concept_explanations.get(c, c.name.lower().replace('_', ' '))
                          for c in frame.concepts]
            parts.append(f"Operations: {', '.join(concept_strs)}")

        if frame.entities:
            entity_strs = [f"{k}={v}" for k, v in frame.entities.items()]
            parts.append(f"Entities: {', '.join(entity_strs)}")

        return " | ".join(parts)

    def suggest_alternatives(self, frame: SemanticFrame) -> List[str]:
        """Suggest alternative commands based on semantic similarity."""
        suggestions = []

        # Based on intent, suggest related commands
        intent_suggestions = {
            IntentType.EXPLORE: ['ls', 'pwd', 'cat'],
            IntentType.QUERY: ['uname', 'whoami', 'date', 'hostname'],
            IntentType.CREATE: ['echo', 'touch', 'mkdir'],
            IntentType.DELETE: ['rm', 'rmdir'],
        }

        if frame.intent in intent_suggestions:
            suggestions.extend(intent_suggestions[frame.intent])

        return suggestions


# ════════════════════════════════════════════════════════════════════════════════
# MAIN SEMANTIC SYSTEM - Integrates all components
# ════════════════════════════════════════════════════════════════════════════════

class NeuralSemanticSystem:
    """
    The main semantic system that integrates all components.
    Provides unified interface for semantic understanding of the OS.
    """

    def __init__(self):
        print("[Semantic System] Initializing Neural Semantic System...")

        # Core components
        self.embedding_network = SemanticEmbeddingNetwork().to(device)
        self.syscall_mapper = SyscallSemanticMapper()
        self.command_analyzer = CommandSemanticAnalyzer()
        self.semantic_memory = SemanticMemory()
        self.nl_interface = NaturalLanguageInterface()
        self.reasoning_engine = SemanticReasoningEngine()

        # Statistics
        self.total_analyses = 0
        self.concepts_detected = defaultdict(int)
        self.intents_detected = defaultdict(int)

        print("[Semantic System] Ready with full semantic understanding!")

    def understand_command(self, command: str, args: List[str] = None) -> SemanticFrame:
        """Understand the semantic meaning of a command."""
        frame = self.command_analyzer.analyze_command(command, args)

        # Enrich with reasoning
        frame.concepts = self.reasoning_engine.infer_implications(frame.concepts)

        # Store in memory
        self.semantic_memory.store_command(command, frame)

        # Update stats
        self.total_analyses += 1
        for concept in frame.concepts:
            self.concepts_detected[concept.name] += 1
        self.intents_detected[frame.intent.name] += 1

        return frame

    def understand_syscalls(self, syscalls: List[int]) -> SemanticFrame:
        """Understand the semantic meaning of a syscall sequence."""
        return self.syscall_mapper.analyze_syscall_sequence(syscalls)

    def understand_natural_language(self, query: str) -> Tuple[SemanticFrame, Optional[str]]:
        """
        Understand natural language query.
        Returns semantic frame and suggested command (if any).
        """
        frame = self.nl_interface.parse_query(query)
        command = self.nl_interface.query_to_command(query)

        cmd_str = command[0] if command else None
        return frame, cmd_str

    def explain(self, frame: SemanticFrame) -> str:
        """Get human-readable explanation of a semantic frame."""
        return self.reasoning_engine.explain_behavior(frame)

    def suggest_next(self, current_frame: SemanticFrame) -> List[str]:
        """Suggest next commands based on current semantic context."""
        return self.reasoning_engine.suggest_alternatives(current_frame)

    def find_similar_commands(self, frame: SemanticFrame) -> List[Tuple[str, float]]:
        """Find commands with similar semantics."""
        return self.semantic_memory.retrieve_similar(frame)

    def get_stats(self) -> Dict:
        """Get system statistics."""
        return {
            'total_analyses': self.total_analyses,
            'concepts_detected': dict(self.concepts_detected),
            'intents_detected': dict(self.intents_detected),
            'commands_in_memory': len(self.semantic_memory.command_memories),
        }


# ════════════════════════════════════════════════════════════════════════════════
# DEMO
# ════════════════════════════════════════════════════════════════════════════════

def demo():
    """Demonstrate the Neural Semantic System."""

    print("=" * 70)
    print("           NEURAL SEMANTIC SYSTEM DEMONSTRATION")
    print("=" * 70)
    print()

    # Initialize
    sem = NeuralSemanticSystem()
    print()

    # Test command understanding
    print("═" * 50)
    print("  COMMAND SEMANTIC ANALYSIS")
    print("═" * 50)

    test_commands = [
        ('ls', []),
        ('cat', ['file.txt']),
        ('echo', ['Hello', 'World']),
        ('grep', ['pattern', 'file.txt']),
        ('uname', ['-a']),
        ('seq', ['1', '5']),
        ('banner', []),
    ]

    for cmd, args in test_commands:
        frame = sem.understand_command(cmd, args)
        explanation = sem.explain(frame)
        print(f"\n  Command: {cmd} {' '.join(args)}")
        print(f"    {explanation}")
        print(f"    Confidence: {frame.confidence:.0%}")

    # Test natural language understanding
    print("\n" + "═" * 50)
    print("  NATURAL LANGUAGE UNDERSTANDING")
    print("═" * 50)

    test_queries = [
        "show me the files",
        "what is the current directory",
        "who am I logged in as",
        "display system info",
        "count to 5",
        "what time is it",
    ]

    for query in test_queries:
        frame, command = sem.understand_natural_language(query)
        print(f"\n  Query: \"{query}\"")
        print(f"    Intent: {frame.intent.name}")
        print(f"    Concepts: {[c.name for c in frame.concepts]}")
        if command:
            print(f"    → Suggested command: {command}")

    # Test syscall understanding
    print("\n" + "═" * 50)
    print("  SYSCALL SEQUENCE ANALYSIS")
    print("═" * 50)

    test_syscalls = [
        ([56, 63, 64, 57], "open-read-write-close pattern"),
        ([160, 64, 93], "uname and exit pattern"),
        ([222, 226, 215], "memory allocation pattern"),
    ]

    for syscalls, description in test_syscalls:
        frame = sem.understand_syscalls(syscalls)
        print(f"\n  Syscalls: {syscalls} ({description})")
        print(f"    Intent: {frame.intent.name}")
        print(f"    Concepts: {[c.name for c in frame.concepts]}")

    # Show statistics
    print("\n" + "═" * 50)
    print("  SEMANTIC SYSTEM STATISTICS")
    print("═" * 50)

    stats = sem.get_stats()
    print(f"\n  Total analyses: {stats['total_analyses']}")
    print(f"  Commands in memory: {stats['commands_in_memory']}")
    print(f"\n  Top intents detected:")
    for intent, count in sorted(stats['intents_detected'].items(), key=lambda x: -x[1])[:5]:
        print(f"    {intent}: {count}")
    print(f"\n  Top concepts detected:")
    for concept, count in sorted(stats['concepts_detected'].items(), key=lambda x: -x[1])[:5]:
        print(f"    {concept}: {count}")

    print("\n" + "=" * 70)
    print("  🧠 Neural Semantic System demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo()
