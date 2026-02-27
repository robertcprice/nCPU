"""
MOONSHOT #4: Psychedelic Bisociation Engine

Grok's idea: "Simulate 'psychedelics' via graph scrambling + reconnection.
Temporally distort graphs -> force unnatural analogies.
Humans discover via altered states; replicate computationally."

This module implements creative program discovery through controlled chaos:

1. GRAPH SCRAMBLING: Controlled perturbation of program dependency graphs
   - Preserves some structural invariants while creating novel connections
   - Multiple "altered state" modes with different scrambling intensities

2. BISOCIATION: Connecting unrelated conceptual domains (Koestler's theory)
   - Maps operations from one domain to structurally similar operations in another
   - Discovers that "loop iteration" is structurally similar to "recursion"
   - Finds that "sorting" patterns can map to "arithmetic optimization"

3. ANALOGY DETECTION: Finding structural similarities between programs
   - Graph isomorphism approximation for computational patterns
   - Embedding-based similarity in latent program space
   - Behavioral equivalence detection

4. EMERGENT PATTERN DISCOVERY: Detecting useful patterns in scrambled graphs
   - Post-scramble validation via KVRM execution
   - Fitness-based retention of discovered novelties
   - Building a library of "creative accidents"

Theory: Human creativity often occurs through:
- Defocused attention (seeing connections between unrelated things)
- Temporal distortion (mixing past patterns with present needs)
- Boundary dissolution (merging separate conceptual domains)

We replicate this computationally through structured randomness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import hashlib
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set, Any, Callable
from enum import Enum
from collections import defaultdict
import copy
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from spnc.instruction_set import (
    Program, Instruction, ARM64Opcode, NUM_REGISTERS, MAX_IMMEDIATE,
    make_add, make_sub, make_mul, make_div, make_mov_imm, make_mov_reg,
    make_cmp, make_cmp_imm, make_branch, make_beq, make_bne, make_blt,
    make_bge, make_bgt, make_ble, make_ret, make_nop, make_and, make_or, make_xor,
    make_lsl, make_lsr
)


# =============================================================================
# ALTERED STATE MODES
# =============================================================================

class AlteredState(Enum):
    """
    Different 'mental states' for creative exploration.

    Inspired by research on how psychedelics affect neural connectivity:
    - Default Mode Network dissolution -> reduced self-referential processing
    - Increased entropy -> more diverse neural firing patterns
    - Cross-modal integration -> synesthesia-like blending
    """

    SOBER = "sober"              # Normal operation, minimal scrambling
    MICRODOSE = "microdose"      # Subtle enhancement, mild pattern relaxation
    THRESHOLD = "threshold"      # Noticeable alterations, pattern mixing begins
    MODERATE = "moderate"        # Significant scrambling, cross-domain connections
    STRONG = "strong"            # Heavy scrambling, boundary dissolution
    HEROIC = "heroic"            # Maximum entropy, complete graph reconstruction
    DREAMING = "dreaming"        # Temporal distortion, memory replay mixing


@dataclass
class AlteredStateConfig:
    """Configuration for an altered state mode."""
    state: AlteredState
    scramble_intensity: float      # 0.0 - 1.0, how much to scramble
    temporal_distortion: float     # 0.0 - 1.0, memory mixing amount
    boundary_dissolution: float    # 0.0 - 1.0, cross-domain blending
    entropy_boost: float           # Multiplier for randomness
    pattern_persistence: float     # How much original structure to preserve
    connection_radius: int         # How far to search for new connections
    oscillation_frequency: float   # For cyclic state changes

    @classmethod
    def for_state(cls, state: AlteredState) -> 'AlteredStateConfig':
        """Get default configuration for a state."""
        configs = {
            AlteredState.SOBER: cls(
                state=state,
                scramble_intensity=0.05,
                temporal_distortion=0.0,
                boundary_dissolution=0.0,
                entropy_boost=1.0,
                pattern_persistence=0.95,
                connection_radius=1,
                oscillation_frequency=0.0
            ),
            AlteredState.MICRODOSE: cls(
                state=state,
                scramble_intensity=0.15,
                temporal_distortion=0.1,
                boundary_dissolution=0.1,
                entropy_boost=1.2,
                pattern_persistence=0.85,
                connection_radius=2,
                oscillation_frequency=0.1
            ),
            AlteredState.THRESHOLD: cls(
                state=state,
                scramble_intensity=0.3,
                temporal_distortion=0.2,
                boundary_dissolution=0.25,
                entropy_boost=1.5,
                pattern_persistence=0.7,
                connection_radius=3,
                oscillation_frequency=0.2
            ),
            AlteredState.MODERATE: cls(
                state=state,
                scramble_intensity=0.5,
                temporal_distortion=0.4,
                boundary_dissolution=0.5,
                entropy_boost=2.0,
                pattern_persistence=0.5,
                connection_radius=5,
                oscillation_frequency=0.3
            ),
            AlteredState.STRONG: cls(
                state=state,
                scramble_intensity=0.7,
                temporal_distortion=0.6,
                boundary_dissolution=0.7,
                entropy_boost=3.0,
                pattern_persistence=0.3,
                connection_radius=8,
                oscillation_frequency=0.5
            ),
            AlteredState.HEROIC: cls(
                state=state,
                scramble_intensity=0.9,
                temporal_distortion=0.8,
                boundary_dissolution=0.9,
                entropy_boost=5.0,
                pattern_persistence=0.1,
                connection_radius=15,
                oscillation_frequency=0.8
            ),
            AlteredState.DREAMING: cls(
                state=state,
                scramble_intensity=0.6,
                temporal_distortion=0.95,  # High temporal mixing
                boundary_dissolution=0.4,
                entropy_boost=2.5,
                pattern_persistence=0.4,
                connection_radius=10,
                oscillation_frequency=1.0   # Oscillating state
            ),
        }
        return configs.get(state, configs[AlteredState.SOBER])


# =============================================================================
# PROGRAM GRAPH REPRESENTATION
# =============================================================================

@dataclass
class GraphNode:
    """A node in the program computation graph."""
    id: int
    opcode: ARM64Opcode
    instruction_idx: int
    inputs: List[int]       # IDs of input nodes
    outputs: List[int]      # IDs of output nodes
    register_reads: Set[int]
    register_writes: Set[int]
    is_control_flow: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgramGraph:
    """
    Graph representation of a program for scrambling.

    Nodes represent instructions, edges represent data/control dependencies.
    """
    nodes: Dict[int, GraphNode]
    edges: List[Tuple[int, int, str]]  # (from_id, to_id, edge_type)
    program: Program

    @classmethod
    def from_program(cls, program: Program) -> 'ProgramGraph':
        """Build a dependency graph from a program."""
        nodes = {}
        edges = []

        # Track which instruction last wrote each register
        last_writer: Dict[int, int] = {}

        for idx, instr in enumerate(program.instructions):
            # Create node
            node = GraphNode(
                id=idx,
                opcode=instr.opcode,
                instruction_idx=idx,
                inputs=[],
                outputs=[],
                register_reads=set(),
                register_writes=set(),
                is_control_flow=ARM64Opcode.is_branch(instr.opcode),
                metadata={}
            )

            # Analyze register usage
            if instr.opcode not in [ARM64Opcode.NOP, ARM64Opcode.RET, ARM64Opcode.B]:
                # Most instructions read rn
                if instr.opcode != ARM64Opcode.MOV or not instr.is_immediate:
                    node.register_reads.add(instr.rn)

                # Most instructions read rm (if not immediate)
                if not instr.is_immediate and instr.opcode != ARM64Opcode.MOV:
                    node.register_reads.add(instr.rm)
                elif instr.opcode == ARM64Opcode.MOV and not instr.is_immediate:
                    node.register_reads.add(instr.rm)

                # Most instructions write rd (except CMP)
                if instr.opcode != ARM64Opcode.CMP:
                    node.register_writes.add(instr.rd)

            # Add data dependency edges
            for reg in node.register_reads:
                if reg in last_writer:
                    writer_idx = last_writer[reg]
                    edges.append((writer_idx, idx, 'data'))
                    nodes[writer_idx].outputs.append(idx)
                    node.inputs.append(writer_idx)

            # Update last writer
            for reg in node.register_writes:
                last_writer[reg] = idx

            # Add control flow edges
            if ARM64Opcode.is_branch(instr.opcode):
                target = instr.branch_target
                if 0 <= target < len(program.instructions):
                    edges.append((idx, target, 'control'))

            # Sequential control flow
            if idx > 0 and not ARM64Opcode.is_branch(program.instructions[idx-1].opcode):
                edges.append((idx-1, idx, 'sequence'))

            nodes[idx] = node

        return cls(nodes=nodes, edges=edges, program=program)

    def to_program(self) -> Program:
        """Reconstruct program from graph (after scrambling)."""
        # Sort nodes by topological order
        visited = set()
        order = []

        def visit(node_id):
            if node_id in visited:
                return
            visited.add(node_id)
            node = self.nodes.get(node_id)
            if node:
                for inp in node.inputs:
                    visit(inp)
                order.append(node_id)

        for node_id in self.nodes:
            visit(node_id)

        # Reconstruct instructions
        instructions = []
        for node_id in order:
            node = self.nodes[node_id]
            orig_instr = self.program.instructions[node.instruction_idx]
            instructions.append(orig_instr)

        # Ensure program ends with RET
        if not instructions or instructions[-1].opcode != ARM64Opcode.RET:
            instructions.append(make_ret())

        return Program(
            instructions=instructions,
            input_registers=self.program.input_registers,
            output_register=self.program.output_register
        )

    def get_adjacency_matrix(self) -> np.ndarray:
        """Get adjacency matrix for graph algorithms."""
        n = len(self.nodes)
        adj = np.zeros((n, n))
        for src, dst, _ in self.edges:
            if src < n and dst < n:
                adj[src, dst] = 1
        return adj

    def get_node_features(self) -> np.ndarray:
        """Get feature matrix for nodes."""
        n = len(self.nodes)
        # Features: [opcode_onehot(30), in_degree, out_degree, is_control]
        features = np.zeros((n, 33))
        for node_id, node in self.nodes.items():
            if node_id < n:
                features[node_id, min(node.opcode.value, 29)] = 1
                features[node_id, 30] = len(node.inputs)
                features[node_id, 31] = len(node.outputs)
                features[node_id, 32] = float(node.is_control_flow)
        return features


# =============================================================================
# GRAPH SCRAMBLING ALGORITHMS
# =============================================================================

class GraphScrambler:
    """
    Controlled chaos generator for program graphs.

    Implements various scrambling strategies that preserve some invariants
    while creating novel connections. The key insight is that useful
    discoveries often happen at the edge of chaos - too much randomness
    produces garbage, too little produces nothing new.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.scramble_history: List[Dict[str, Any]] = []

    def scramble(self, graph: ProgramGraph,
                 config: AlteredStateConfig) -> ProgramGraph:
        """
        Apply controlled scrambling to a program graph.

        Returns a new graph with scrambled connections while preserving
        some structural invariants based on the altered state configuration.
        """
        # Deep copy the graph
        new_graph = self._copy_graph(graph)

        # Apply different scrambling techniques based on intensity
        if config.scramble_intensity > 0.1:
            new_graph = self._scramble_edges(new_graph, config)

        if config.scramble_intensity > 0.3:
            new_graph = self._scramble_opcodes(new_graph, config)

        if config.scramble_intensity > 0.5:
            new_graph = self._scramble_registers(new_graph, config)

        if config.temporal_distortion > 0.2:
            new_graph = self._temporal_scramble(new_graph, config)

        if config.boundary_dissolution > 0.3:
            new_graph = self._dissolve_boundaries(new_graph, config)

        # Record what we did
        self.scramble_history.append({
            'config': config,
            'original_nodes': len(graph.nodes),
            'new_nodes': len(new_graph.nodes),
            'original_edges': len(graph.edges),
            'new_edges': len(new_graph.edges)
        })

        return new_graph

    def _copy_graph(self, graph: ProgramGraph) -> ProgramGraph:
        """Deep copy a program graph."""
        new_nodes = {}
        for node_id, node in graph.nodes.items():
            new_nodes[node_id] = GraphNode(
                id=node.id,
                opcode=node.opcode,
                instruction_idx=node.instruction_idx,
                inputs=list(node.inputs),
                outputs=list(node.outputs),
                register_reads=set(node.register_reads),
                register_writes=set(node.register_writes),
                is_control_flow=node.is_control_flow,
                metadata=dict(node.metadata)
            )
        new_edges = list(graph.edges)
        return ProgramGraph(
            nodes=new_nodes,
            edges=new_edges,
            program=graph.program
        )

    def _scramble_edges(self, graph: ProgramGraph,
                        config: AlteredStateConfig) -> ProgramGraph:
        """
        Scramble edge connections while preserving graph properties.

        Strategy: Rewire some edges to different targets within
        the connection radius, preserving in/out degree distribution.
        """
        num_to_scramble = int(len(graph.edges) * config.scramble_intensity)

        if num_to_scramble == 0:
            return graph

        # Select edges to scramble (prefer non-control edges)
        data_edges = [(i, e) for i, e in enumerate(graph.edges)
                      if e[2] == 'data']

        if not data_edges:
            return graph

        indices_to_scramble = self.rng.choice(
            [i for i, _ in data_edges],
            size=min(num_to_scramble, len(data_edges)),
            replace=False
        )

        new_edges = list(graph.edges)
        node_ids = list(graph.nodes.keys())

        for idx in indices_to_scramble:
            src, dst, edge_type = new_edges[idx]

            # Find new destination within connection radius
            candidates = [
                n for n in node_ids
                if n != src and abs(n - dst) <= config.connection_radius
            ]

            if candidates:
                new_dst = self.rng.choice(candidates)
                new_edges[idx] = (src, new_dst, edge_type)

                # Update node inputs/outputs
                if dst in graph.nodes[src].outputs:
                    graph.nodes[src].outputs.remove(dst)
                    graph.nodes[src].outputs.append(new_dst)
                if src in graph.nodes[dst].inputs:
                    graph.nodes[dst].inputs.remove(src)
                if new_dst in graph.nodes:
                    graph.nodes[new_dst].inputs.append(src)

        graph.edges = new_edges
        return graph

    def _scramble_opcodes(self, graph: ProgramGraph,
                          config: AlteredStateConfig) -> ProgramGraph:
        """
        Scramble opcodes while preserving category (arithmetic stays arithmetic).

        This is like seeing addition as multiplication - maintaining the
        "doing math" concept while changing the specific operation.
        """
        # Define opcode categories for structured scrambling
        categories = {
            'arithmetic': [ARM64Opcode.ADD, ARM64Opcode.SUB, ARM64Opcode.MUL, ARM64Opcode.UDIV],
            'logical': [ARM64Opcode.AND, ARM64Opcode.ORR, ARM64Opcode.EOR],
            'shift': [ARM64Opcode.LSL, ARM64Opcode.LSR, ARM64Opcode.ASR],
            'move': [ARM64Opcode.MOV],
            'compare': [ARM64Opcode.CMP],
            'control': [ARM64Opcode.B, ARM64Opcode.BEQ, ARM64Opcode.BNE,
                       ARM64Opcode.BLT, ARM64Opcode.BGE, ARM64Opcode.BGT, ARM64Opcode.BLE],
        }

        # Reverse lookup
        opcode_to_category = {}
        for cat, ops in categories.items():
            for op in ops:
                opcode_to_category[op] = cat

        num_to_scramble = int(len(graph.nodes) * config.scramble_intensity)
        node_ids = list(graph.nodes.keys())

        if num_to_scramble > 0 and node_ids:
            indices_to_scramble = self.rng.choice(
                node_ids,
                size=min(num_to_scramble, len(node_ids)),
                replace=False
            )

            for node_id in indices_to_scramble:
                node = graph.nodes[node_id]

                # Skip control flow and special instructions
                if node.opcode in [ARM64Opcode.RET, ARM64Opcode.NOP]:
                    continue

                category = opcode_to_category.get(node.opcode)
                if category and category in categories:
                    # Chance to stay in category vs cross categories
                    if self.rng.random() < config.boundary_dissolution:
                        # Cross-category scramble
                        all_ops = [op for ops in categories.values() for op in ops]
                        new_opcode = self.rng.choice(all_ops)
                    else:
                        # Within-category scramble
                        new_opcode = self.rng.choice(categories[category])

                    node.opcode = new_opcode

        return graph

    def _scramble_registers(self, graph: ProgramGraph,
                            config: AlteredStateConfig) -> ProgramGraph:
        """
        Scramble register assignments while maintaining data flow.

        Like renaming variables in your mind - the logic stays the same
        but the symbols are shuffled.
        """
        # Build register mapping
        used_registers = set()
        for node in graph.nodes.values():
            used_registers.update(node.register_reads)
            used_registers.update(node.register_writes)

        if not used_registers:
            return graph

        # Create permutation with some fixed points (based on persistence)
        used_list = list(used_registers)
        mapping = {}

        for reg in used_list:
            if self.rng.random() < config.pattern_persistence:
                # Keep original
                mapping[reg] = reg
            else:
                # Map to different register in range 0-7 (commonly used)
                mapping[reg] = self.rng.randint(0, 8)

        # Apply mapping to nodes
        for node in graph.nodes.values():
            node.register_reads = {mapping.get(r, r) for r in node.register_reads}
            node.register_writes = {mapping.get(r, r) for r in node.register_writes}

        return graph

    def _temporal_scramble(self, graph: ProgramGraph,
                           config: AlteredStateConfig) -> ProgramGraph:
        """
        Scramble temporal order of instructions.

        Like memories mixing together - instructions from different
        points in the program blend into new sequences.
        """
        node_ids = list(graph.nodes.keys())
        n = len(node_ids)

        if n < 3:
            return graph

        # Number of swaps based on temporal distortion
        num_swaps = int(n * config.temporal_distortion)

        for _ in range(num_swaps):
            # Select two non-adjacent nodes to swap
            i = self.rng.randint(0, n)
            j = self.rng.randint(0, n)

            if i != j and abs(i - j) > 1:
                # Swap instruction indices
                node_i = graph.nodes.get(node_ids[i])
                node_j = graph.nodes.get(node_ids[j])

                if node_i and node_j:
                    # Don't swap if one is RET
                    if node_i.opcode == ARM64Opcode.RET or node_j.opcode == ARM64Opcode.RET:
                        continue

                    # Swap instruction indices
                    node_i.instruction_idx, node_j.instruction_idx = \
                        node_j.instruction_idx, node_i.instruction_idx

        return graph

    def _dissolve_boundaries(self, graph: ProgramGraph,
                             config: AlteredStateConfig) -> ProgramGraph:
        """
        Dissolve boundaries between different parts of the graph.

        Like ego dissolution - the sense of separate "modules" in the
        program merges into a unified whole with unexpected connections.
        """
        # Identify "boundaries" - places where there's no direct connection
        # between temporally adjacent instructions
        adjacency = graph.get_adjacency_matrix()
        n = adjacency.shape[0]

        # Find gaps in connectivity
        gaps = []
        for i in range(n - 1):
            if adjacency[i, i+1] == 0 and adjacency[i+1, i] == 0:
                gaps.append((i, i+1))

        # Dissolve some gaps by adding new connections
        num_to_dissolve = int(len(gaps) * config.boundary_dissolution)

        if num_to_dissolve > 0 and gaps:
            indices = self.rng.choice(
                len(gaps),
                size=min(num_to_dissolve, len(gaps)),
                replace=False
            )

            for idx in indices:
                i, j = gaps[idx]
                # Add a data edge
                graph.edges.append((i, j, 'data'))
                if i in graph.nodes:
                    graph.nodes[i].outputs.append(j)
                if j in graph.nodes:
                    graph.nodes[j].inputs.append(i)

        return graph


# =============================================================================
# BISOCIATION ENGINE
# =============================================================================

@dataclass
class ConceptualDomain:
    """A domain of related computational concepts."""
    name: str
    opcodes: List[ARM64Opcode]
    patterns: List[List[ARM64Opcode]]  # Common opcode sequences
    semantic_features: Dict[str, float]  # Feature vector for similarity


class BisociationEngine:
    """
    Implements Koestler's theory of bisociation for program synthesis.

    Bisociation: The creative act of connecting two previously unrelated
    matrices of thought (conceptual domains) to produce novel insights.

    Examples:
    - Gutenberg bisociated wine press + coin punch -> printing press
    - Darwin bisociated artificial selection + natural variation -> evolution
    - We bisociate arithmetic patterns + control flow -> novel algorithms
    """

    def __init__(self):
        self.domains = self._initialize_domains()
        self.analogy_cache: Dict[str, List[Tuple[Any, float]]] = {}
        self.discovered_bisociations: List[Dict[str, Any]] = []

    def _initialize_domains(self) -> Dict[str, ConceptualDomain]:
        """Initialize computational concept domains."""
        return {
            'arithmetic': ConceptualDomain(
                name='arithmetic',
                opcodes=[ARM64Opcode.ADD, ARM64Opcode.SUB, ARM64Opcode.MUL, ARM64Opcode.UDIV],
                patterns=[
                    [ARM64Opcode.ADD, ARM64Opcode.ADD],  # Double
                    [ARM64Opcode.MUL, ARM64Opcode.ADD],  # Polynomial
                    [ARM64Opcode.SUB, ARM64Opcode.MUL],  # Difference of squares potential
                ],
                semantic_features={
                    'numerical': 1.0,
                    'transformative': 0.8,
                    'invertible': 0.7,
                    'commutative': 0.6,
                    'associative': 0.6,
                }
            ),
            'logical': ConceptualDomain(
                name='logical',
                opcodes=[ARM64Opcode.AND, ARM64Opcode.ORR, ARM64Opcode.EOR],
                patterns=[
                    [ARM64Opcode.AND, ARM64Opcode.ORR],  # Masking
                    [ARM64Opcode.EOR, ARM64Opcode.EOR],  # XOR swap
                    [ARM64Opcode.AND, ARM64Opcode.EOR],  # Bit manipulation
                ],
                semantic_features={
                    'numerical': 0.3,
                    'transformative': 0.9,
                    'invertible': 0.8,
                    'commutative': 0.7,
                    'associative': 0.7,
                }
            ),
            'shift': ConceptualDomain(
                name='shift',
                opcodes=[ARM64Opcode.LSL, ARM64Opcode.LSR],
                patterns=[
                    [ARM64Opcode.LSL, ARM64Opcode.LSL],  # Power of 4
                    [ARM64Opcode.LSR, ARM64Opcode.AND],  # Extract bits
                ],
                semantic_features={
                    'numerical': 0.6,
                    'transformative': 0.9,
                    'invertible': 0.5,
                    'commutative': 0.0,
                    'associative': 0.0,
                }
            ),
            'control': ConceptualDomain(
                name='control',
                opcodes=[ARM64Opcode.CMP, ARM64Opcode.BEQ, ARM64Opcode.BNE,
                        ARM64Opcode.BLT, ARM64Opcode.BGE],
                patterns=[
                    [ARM64Opcode.CMP, ARM64Opcode.BEQ],  # Equality check
                    [ARM64Opcode.CMP, ARM64Opcode.BLT],  # Less than check
                    [ARM64Opcode.CMP, ARM64Opcode.BNE, ARM64Opcode.ADD],  # Loop pattern
                ],
                semantic_features={
                    'numerical': 0.2,
                    'transformative': 0.3,
                    'invertible': 0.4,
                    'commutative': 0.0,
                    'associative': 0.0,
                }
            ),
            'memory': ConceptualDomain(
                name='memory',
                opcodes=[ARM64Opcode.MOV],
                patterns=[
                    [ARM64Opcode.MOV, ARM64Opcode.MOV],  # Copy chain
                    [ARM64Opcode.MOV, ARM64Opcode.ADD, ARM64Opcode.MOV],  # Accumulate
                ],
                semantic_features={
                    'numerical': 0.1,
                    'transformative': 0.5,
                    'invertible': 1.0,
                    'commutative': 0.0,
                    'associative': 0.0,
                }
            ),
        }

    def bisociate(self, program: Program,
                  source_domain: str,
                  target_domain: str,
                  config: AlteredStateConfig) -> List[Program]:
        """
        Create new programs by bisociating concepts from two domains.

        Args:
            program: Source program to transform
            source_domain: Original conceptual domain
            target_domain: Domain to map concepts to
            config: Altered state configuration

        Returns:
            List of new programs created through bisociation
        """
        if source_domain not in self.domains or target_domain not in self.domains:
            return []

        source = self.domains[source_domain]
        target = self.domains[target_domain]

        # Find mapping between domains based on semantic similarity
        opcode_mapping = self._find_opcode_mapping(source, target, config)

        # Apply mapping to create new programs
        results = []

        for _ in range(int(3 * config.entropy_boost)):
            new_instructions = []

            for instr in program.instructions:
                if instr.opcode in opcode_mapping:
                    # Map to target domain opcode
                    if random.random() < config.scramble_intensity:
                        new_opcode = opcode_mapping[instr.opcode]
                        new_instr = Instruction(
                            opcode=new_opcode,
                            rd=instr.rd,
                            rn=instr.rn,
                            rm=instr.rm,
                            is_immediate=instr.is_immediate,
                            branch_target=instr.branch_target
                        )
                        new_instructions.append(new_instr)
                    else:
                        new_instructions.append(instr)
                else:
                    new_instructions.append(instr)

            if new_instructions and new_instructions[-1].opcode != ARM64Opcode.RET:
                new_instructions.append(make_ret())

            new_program = Program(
                instructions=new_instructions,
                input_registers=program.input_registers,
                output_register=program.output_register
            )

            # Record the bisociation
            self.discovered_bisociations.append({
                'source_domain': source_domain,
                'target_domain': target_domain,
                'mapping': opcode_mapping,
                'original': program,
                'result': new_program,
            })

            results.append(new_program)

        return results

    def _find_opcode_mapping(self, source: ConceptualDomain,
                             target: ConceptualDomain,
                             config: AlteredStateConfig) -> Dict[ARM64Opcode, ARM64Opcode]:
        """
        Find mapping between opcodes based on semantic similarity.

        Uses a combination of:
        - Structural similarity (same position in domain)
        - Semantic feature similarity
        - Random exploration based on entropy boost
        """
        mapping = {}

        for i, src_op in enumerate(source.opcodes):
            if i < len(target.opcodes):
                # Direct structural mapping
                if random.random() < config.pattern_persistence:
                    mapping[src_op] = target.opcodes[i]
                else:
                    # Random mapping within target domain
                    mapping[src_op] = random.choice(target.opcodes)
            else:
                # Wrap around if target has fewer opcodes
                mapping[src_op] = target.opcodes[i % len(target.opcodes)]

        return mapping

    def find_analogies(self, program: Program,
                       library: List[Program],
                       top_k: int = 5) -> List[Tuple[Program, float]]:
        """
        Find programs in the library that are analogous to the input.

        Analogy is defined as structural similarity with different surface form.
        Like "atom is to nucleus as solar system is to sun."
        """
        query_graph = ProgramGraph.from_program(program)
        query_features = query_graph.get_node_features()
        query_adj = query_graph.get_adjacency_matrix()

        similarities = []

        for lib_prog in library:
            lib_graph = ProgramGraph.from_program(lib_prog)
            lib_features = lib_graph.get_node_features()
            lib_adj = lib_graph.get_adjacency_matrix()

            # Compute similarity
            similarity = self._compute_graph_similarity(
                query_features, query_adj,
                lib_features, lib_adj
            )

            similarities.append((lib_prog, similarity))

        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _compute_graph_similarity(self,
                                   feat1: np.ndarray, adj1: np.ndarray,
                                   feat2: np.ndarray, adj2: np.ndarray) -> float:
        """
        Compute similarity between two program graphs.

        Uses a combination of:
        - Node feature similarity (what operations are used)
        - Structural similarity (how they're connected)
        - Size similarity penalty
        """
        # Size similarity
        size_sim = 1.0 - abs(len(feat1) - len(feat2)) / max(len(feat1), len(feat2), 1)

        # Feature similarity (average cosine similarity of node features)
        if len(feat1) == 0 or len(feat2) == 0:
            return 0.0

        feat_sim = 0.0
        for f1 in feat1:
            max_sim = 0.0
            for f2 in feat2:
                norm1 = np.linalg.norm(f1)
                norm2 = np.linalg.norm(f2)
                if norm1 > 0 and norm2 > 0:
                    cos_sim = np.dot(f1, f2) / (norm1 * norm2)
                    max_sim = max(max_sim, cos_sim)
            feat_sim += max_sim
        feat_sim /= len(feat1)

        # Structural similarity (graph spectrum comparison)
        struct_sim = self._spectral_similarity(adj1, adj2)

        # Weighted combination
        total_sim = 0.3 * size_sim + 0.4 * feat_sim + 0.3 * struct_sim
        return total_sim

    def _spectral_similarity(self, adj1: np.ndarray, adj2: np.ndarray) -> float:
        """
        Compare graph structure using spectral properties.

        The eigenvalues of the adjacency matrix capture structural properties
        like connectivity, cycles, and path lengths.
        """
        try:
            # Get eigenvalues (sorted by magnitude)
            eig1 = np.sort(np.abs(np.linalg.eigvals(adj1)))
            eig2 = np.sort(np.abs(np.linalg.eigvals(adj2)))

            # Pad to same length
            max_len = max(len(eig1), len(eig2))
            eig1 = np.pad(eig1, (0, max_len - len(eig1)))
            eig2 = np.pad(eig2, (0, max_len - len(eig2)))

            # Compute similarity (inverse of L2 distance)
            dist = np.linalg.norm(eig1 - eig2)
            return 1.0 / (1.0 + dist)
        except:
            return 0.5  # Default similarity on error


# =============================================================================
# EMERGENT PATTERN DISCOVERY
# =============================================================================

@dataclass
class EmergentPattern:
    """A pattern discovered through scrambling that proved useful."""
    pattern_id: str
    original_program: Program
    scrambled_program: Program
    config_used: AlteredStateConfig
    fitness_improvement: float
    discovery_time: float
    opcode_sequence: List[ARM64Opcode]
    is_novel: bool


class PatternDiscoveryEngine:
    """
    Discovers useful patterns that emerge from controlled chaos.

    The key insight is that most scrambling produces garbage, but
    occasionally it produces something useful. We need to:
    1. Generate many scrambled variants
    2. Evaluate them for useful behavior
    3. Extract the successful patterns
    4. Build a library of "creative accidents"
    """

    def __init__(self, evaluator: Optional[Callable] = None):
        self.scrambler = GraphScrambler()
        self.bisociator = BisociationEngine()
        self.discovered_patterns: List[EmergentPattern] = []
        self.pattern_library: Dict[str, EmergentPattern] = {}
        self.evaluator = evaluator or self._default_evaluator

    def _default_evaluator(self, program: Program,
                           test_cases: List[Tuple[int, int]]) -> float:
        """Default fitness evaluator."""
        try:
            correct = 0
            for inp, expected in test_cases:
                result = self._execute_program(program, inp)
                if result == expected:
                    correct += 1
            return correct / len(test_cases) if test_cases else 0.0
        except:
            return 0.0

    def _execute_program(self, program: Program, input_val: int) -> int:
        """Simple program executor."""
        registers = [0] * 32
        registers[0] = input_val
        flags = {'N': 0, 'Z': 0, 'C': 0, 'V': 0}

        pc = 0
        max_cycles = 100
        MAX_VAL = (1 << 64) - 1

        while pc < len(program.instructions) and max_cycles > 0:
            instr = program.instructions[pc]
            max_cycles -= 1

            if instr.opcode == ARM64Opcode.RET:
                break
            elif instr.opcode == ARM64Opcode.NOP:
                pc += 1
                continue

            operand = instr.rm if instr.is_immediate else registers[min(instr.rm, 31)]

            if instr.opcode == ARM64Opcode.ADD:
                registers[instr.rd] = (registers[instr.rn] + operand) & MAX_VAL
            elif instr.opcode == ARM64Opcode.SUB:
                registers[instr.rd] = (registers[instr.rn] - operand) & MAX_VAL
            elif instr.opcode == ARM64Opcode.MUL:
                registers[instr.rd] = (registers[instr.rn] * operand) & MAX_VAL
            elif instr.opcode == ARM64Opcode.UDIV:
                registers[instr.rd] = registers[instr.rn] // operand if operand else 0
            elif instr.opcode == ARM64Opcode.AND:
                registers[instr.rd] = registers[instr.rn] & operand
            elif instr.opcode == ARM64Opcode.ORR:
                registers[instr.rd] = registers[instr.rn] | operand
            elif instr.opcode == ARM64Opcode.EOR:
                registers[instr.rd] = registers[instr.rn] ^ operand
            elif instr.opcode == ARM64Opcode.MOV:
                registers[instr.rd] = operand
            elif instr.opcode == ARM64Opcode.LSL:
                shift = operand & 63
                registers[instr.rd] = (registers[instr.rn] << shift) & MAX_VAL
            elif instr.opcode == ARM64Opcode.LSR:
                shift = operand & 63
                registers[instr.rd] = registers[instr.rn] >> shift
            elif instr.opcode == ARM64Opcode.CMP:
                diff = registers[instr.rn] - operand
                flags['N'] = 1 if diff < 0 else 0
                flags['Z'] = 1 if diff == 0 else 0
            elif ARM64Opcode.is_branch(instr.opcode):
                should_branch = False
                if instr.opcode == ARM64Opcode.B:
                    should_branch = True
                elif instr.opcode == ARM64Opcode.BEQ and flags['Z'] == 1:
                    should_branch = True
                elif instr.opcode == ARM64Opcode.BNE and flags['Z'] == 0:
                    should_branch = True

                if should_branch:
                    pc = instr.branch_target
                    continue

            pc += 1

        return registers[program.output_register]

    def explore(self, seed_programs: List[Program],
                test_cases: List[Tuple[int, int]],
                state: AlteredState = AlteredState.MODERATE,
                iterations: int = 100) -> List[EmergentPattern]:
        """
        Explore program space through altered-state scrambling.

        Args:
            seed_programs: Starting programs to scramble
            test_cases: Test cases to evaluate fitness
            state: Altered state mode to use
            iterations: Number of scrambling iterations

        Returns:
            List of useful patterns discovered
        """
        config = AlteredStateConfig.for_state(state)
        discoveries = []

        import time
        start_time = time.time()

        for iteration in range(iterations):
            # Select a seed program
            seed = random.choice(seed_programs)
            original_fitness = self.evaluator(seed, test_cases)

            # Build graph and scramble
            graph = ProgramGraph.from_program(seed)
            scrambled_graph = self.scrambler.scramble(graph, config)

            # Reconstruct program
            try:
                scrambled_program = scrambled_graph.to_program()

                # Evaluate scrambled program
                scrambled_fitness = self.evaluator(scrambled_program, test_cases)

                # Check if improvement
                improvement = scrambled_fitness - original_fitness

                if improvement > 0.1:  # Significant improvement
                    pattern = EmergentPattern(
                        pattern_id=f"emergent_{len(self.discovered_patterns)}",
                        original_program=seed,
                        scrambled_program=scrambled_program,
                        config_used=config,
                        fitness_improvement=improvement,
                        discovery_time=time.time() - start_time,
                        opcode_sequence=[i.opcode for i in scrambled_program.instructions],
                        is_novel=self._is_novel_pattern(scrambled_program)
                    )
                    discoveries.append(pattern)
                    self.discovered_patterns.append(pattern)
                    self.pattern_library[pattern.pattern_id] = pattern

                    print(f"  [*] Discovered pattern! Improvement: {improvement:.3f}")

            except Exception as e:
                # Invalid program generated - continue
                continue

            # Also try bisociation
            if random.random() < 0.3:
                domains = list(self.bisociator.domains.keys())
                if len(domains) >= 2:
                    src, tgt = random.sample(domains, 2)
                    bisociated = self.bisociator.bisociate(seed, src, tgt, config)

                    for bis_prog in bisociated:
                        try:
                            bis_fitness = self.evaluator(bis_prog, test_cases)
                            bis_improvement = bis_fitness - original_fitness

                            if bis_improvement > 0.1:
                                pattern = EmergentPattern(
                                    pattern_id=f"bisociate_{len(self.discovered_patterns)}",
                                    original_program=seed,
                                    scrambled_program=bis_prog,
                                    config_used=config,
                                    fitness_improvement=bis_improvement,
                                    discovery_time=time.time() - start_time,
                                    opcode_sequence=[i.opcode for i in bis_prog.instructions],
                                    is_novel=True
                                )
                                discoveries.append(pattern)
                                self.discovered_patterns.append(pattern)

                                print(f"  [*] Bisociation discovery! {src}->{tgt}, Improvement: {bis_improvement:.3f}")
                        except:
                            continue

            # Progress report
            if (iteration + 1) % 20 == 0:
                print(f"  Iteration {iteration + 1}/{iterations}, Discoveries: {len(discoveries)}")

        return discoveries

    def _is_novel_pattern(self, program: Program) -> bool:
        """Check if program represents a novel pattern."""
        opcode_seq = tuple(i.opcode for i in program.instructions)

        for existing in self.discovered_patterns:
            existing_seq = tuple(existing.opcode_sequence)
            if opcode_seq == existing_seq:
                return False

        return True


# =============================================================================
# MAIN BISOCIATION ENGINE
# =============================================================================

class PsychedelicBisociationEngine:
    """
    The main engine combining all components for creative program discovery.

    This is the unified interface for:
    1. Altered state management
    2. Graph scrambling
    3. Bisociation across domains
    4. Pattern discovery and retention
    5. KVRM validation of discoveries
    """

    def __init__(self, kvrm_executor: Optional[Any] = None):
        self.scrambler = GraphScrambler()
        self.bisociator = BisociationEngine()
        self.pattern_discovery = PatternDiscoveryEngine()
        self.kvrm_executor = kvrm_executor

        # State management
        self.current_state = AlteredState.SOBER
        self.state_history: List[Tuple[AlteredState, float]] = []

        # Discovery tracking
        self.total_explorations = 0
        self.successful_discoveries = 0
        self.validation_cache: Dict[str, bool] = {}

    def set_state(self, state: AlteredState):
        """Change the altered state mode."""
        self.current_state = state
        import time
        self.state_history.append((state, time.time()))
        print(f"State changed to: {state.value}")

    def create_trip(self, seed_programs: List[Program],
                    test_cases: List[Tuple[int, int]],
                    duration: int = 100,
                    peak_state: AlteredState = AlteredState.MODERATE) -> Dict[str, Any]:
        """
        A complete "trip" through altered states for program discovery.

        Follows a typical psychedelic experience arc:
        1. Come-up: SOBER -> THRESHOLD
        2. Peak: THRESHOLD -> peak_state
        3. Plateau: Hold at peak
        4. Come-down: peak_state -> SOBER

        Args:
            seed_programs: Programs to use as starting material
            test_cases: Evaluation criteria
            duration: Total iterations
            peak_state: Maximum intensity state to reach

        Returns:
            Dictionary with trip results and discoveries
        """
        import time
        start_time = time.time()

        print("\n" + "="*70)
        print("  PSYCHEDELIC BISOCIATION ENGINE - TRIP INITIATED")
        print("="*70)
        print(f"  Seed programs: {len(seed_programs)}")
        print(f"  Test cases: {len(test_cases)}")
        print(f"  Duration: {duration} iterations")
        print(f"  Peak state: {peak_state.value}")
        print("="*70 + "\n")

        discoveries = []
        best_fitness = 0.0
        best_program = None

        # Define state sequence (trip arc)
        state_sequence = self._generate_trip_arc(duration, peak_state)

        for iteration, state in enumerate(state_sequence):
            self.set_state(state)
            config = AlteredStateConfig.for_state(state)

            # Select seed
            seed = random.choice(seed_programs)

            # Scramble
            graph = ProgramGraph.from_program(seed)
            scrambled_graph = self.scrambler.scramble(graph, config)

            try:
                scrambled_program = scrambled_graph.to_program()

                # Evaluate
                fitness = self.pattern_discovery.evaluator(scrambled_program, test_cases)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_program = scrambled_program

                    print(f"  [{state.value}] New best: {fitness:.3f}")

                    if fitness >= 1.0:
                        # Perfect solution found!
                        print(f"\n  [!!!] PERFECT SOLUTION FOUND!")
                        break

                # Try bisociation at higher states
                if config.scramble_intensity > 0.3:
                    domains = list(self.bisociator.domains.keys())
                    src, tgt = random.sample(domains, 2)
                    bisociated = self.bisociator.bisociate(seed, src, tgt, config)

                    for bis_prog in bisociated:
                        bis_fitness = self.pattern_discovery.evaluator(bis_prog, test_cases)
                        if bis_fitness > best_fitness:
                            best_fitness = bis_fitness
                            best_program = bis_prog
                            print(f"  [{state.value}] Bisociation improvement: {bis_fitness:.3f}")

            except Exception as e:
                continue

            self.total_explorations += 1

            # Progress
            if (iteration + 1) % (duration // 5) == 0:
                elapsed = time.time() - start_time
                print(f"\n  Progress: {iteration + 1}/{duration} ({elapsed:.1f}s)")
                print(f"  Best fitness: {best_fitness:.3f}")
                print(f"  Current state: {state.value}\n")

        # Validate best program with KVRM if available
        kvrm_validated = False
        if self.kvrm_executor and best_program:
            kvrm_validated = self._validate_with_kvrm(best_program, test_cases)

        if best_fitness > 0:
            self.successful_discoveries += 1

        results = {
            'success': best_fitness >= 1.0,
            'best_program': best_program,
            'best_fitness': best_fitness,
            'kvrm_validated': kvrm_validated,
            'iterations': len(state_sequence),
            'time_seconds': time.time() - start_time,
            'state_history': self.state_history[-len(state_sequence):],
            'discoveries': discoveries,
        }

        print("\n" + "="*70)
        print("  TRIP COMPLETE")
        print("="*70)
        print(f"  Success: {results['success']}")
        print(f"  Best fitness: {results['best_fitness']:.3f}")
        print(f"  KVRM validated: {results['kvrm_validated']}")
        print(f"  Time: {results['time_seconds']:.2f}s")
        print("="*70 + "\n")

        return results

    def _generate_trip_arc(self, duration: int,
                           peak_state: AlteredState) -> List[AlteredState]:
        """Generate the state sequence for a trip."""
        states = list(AlteredState)
        peak_idx = states.index(peak_state)

        sequence = []

        # Come-up (20% of duration)
        comeup_length = duration // 5
        for i in range(comeup_length):
            progress = i / comeup_length
            state_idx = int(progress * peak_idx)
            sequence.append(states[min(state_idx, len(states) - 1)])

        # Peak (40% of duration)
        peak_length = 2 * duration // 5
        for _ in range(peak_length):
            sequence.append(peak_state)

        # Come-down (40% of duration)
        comedown_length = duration - len(sequence)
        for i in range(comedown_length):
            progress = i / comedown_length
            state_idx = int((1 - progress) * peak_idx)
            sequence.append(states[max(state_idx, 0)])

        return sequence

    def _validate_with_kvrm(self, program: Program,
                            test_cases: List[Tuple[int, int]]) -> bool:
        """Validate a program using the KVRM executor."""
        if not self.kvrm_executor:
            return False

        try:
            for inp, expected in test_cases:
                result = self.kvrm_executor.execute(program, inp)
                if result != expected:
                    return False
            return True
        except:
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            'total_explorations': self.total_explorations,
            'successful_discoveries': self.successful_discoveries,
            'discovery_rate': self.successful_discoveries / max(1, self.total_explorations),
            'patterns_discovered': len(self.pattern_discovery.discovered_patterns),
            'bisociations_found': len(self.bisociator.discovered_bisociations),
            'current_state': self.current_state.value,
        }


# =============================================================================
# TEST AND DEMONSTRATION
# =============================================================================

def create_seed_programs() -> List[Program]:
    """Create a library of seed programs for exploration."""
    seeds = []

    # Identity: return input
    seeds.append(Program(
        instructions=[make_ret()],
        input_registers=[0],
        output_register=0
    ))

    # Double: x0 = x0 + x0
    seeds.append(Program(
        instructions=[
            make_add(0, 0, 0),
            make_ret()
        ],
        input_registers=[0],
        output_register=0
    ))

    # Increment: x0 = x0 + 1
    seeds.append(Program(
        instructions=[
            Instruction(ARM64Opcode.ADD, rd=0, rn=0, rm=1, is_immediate=True),
            make_ret()
        ],
        input_registers=[0],
        output_register=0
    ))

    # Triple: x0 = x0 + x0 + x0
    seeds.append(Program(
        instructions=[
            make_mov_reg(1, 0),
            make_add(0, 0, 0),
            make_add(0, 0, 1),
            make_ret()
        ],
        input_registers=[0],
        output_register=0
    ))

    # Square: x0 = x0 * x0
    seeds.append(Program(
        instructions=[
            make_mul(0, 0, 0),
            make_ret()
        ],
        input_registers=[0],
        output_register=0
    ))

    # Quadruple: x0 = 4 * x0
    seeds.append(Program(
        instructions=[
            make_add(0, 0, 0),
            make_add(0, 0, 0),
            make_ret()
        ],
        input_registers=[0],
        output_register=0
    ))

    # Multiply by 3 via shifts and add
    seeds.append(Program(
        instructions=[
            make_mov_reg(1, 0),
            make_lsl(0, 0, 1),  # x0 = x0 * 2
            make_add(0, 0, 1),  # x0 = x0 + x1 = 3 * original
            make_ret()
        ],
        input_registers=[0],
        output_register=0
    ))

    # XOR swap pattern
    seeds.append(Program(
        instructions=[
            make_xor(0, 0, 1),
            make_xor(1, 1, 0),
            make_xor(0, 0, 1),
            make_ret()
        ],
        input_registers=[0, 1],
        output_register=0
    ))

    return seeds


def demo_basic_scrambling():
    """Demonstrate basic graph scrambling."""
    print("\n" + "="*70)
    print("  DEMO: Basic Graph Scrambling")
    print("="*70)

    # Create a simple program
    program = Program(
        instructions=[
            make_add(0, 0, 0),
            make_add(0, 0, 0),
            make_ret()
        ],
        input_registers=[0],
        output_register=0
    )

    print("\nOriginal program:")
    print(program)

    # Build graph
    graph = ProgramGraph.from_program(program)
    print(f"\nGraph has {len(graph.nodes)} nodes and {len(graph.edges)} edges")

    # Scramble at different intensities
    scrambler = GraphScrambler()

    for state in [AlteredState.MICRODOSE, AlteredState.MODERATE, AlteredState.HEROIC]:
        config = AlteredStateConfig.for_state(state)
        scrambled = scrambler.scramble(graph, config)

        print(f"\n[{state.value}] Scrambled program:")
        try:
            new_prog = scrambled.to_program()
            for i, instr in enumerate(new_prog.instructions):
                print(f"  {i}: {instr}")
        except Exception as e:
            print(f"  Error reconstructing: {e}")


def demo_bisociation():
    """Demonstrate cross-domain bisociation."""
    print("\n" + "="*70)
    print("  DEMO: Cross-Domain Bisociation")
    print("="*70)

    engine = BisociationEngine()

    # Start with an arithmetic program
    program = Program(
        instructions=[
            make_add(0, 0, 0),
            make_mul(0, 0, 1),
            make_ret()
        ],
        input_registers=[0, 1],
        output_register=0
    )

    print("\nOriginal (arithmetic) program:")
    print(program)

    # Bisociate to logical domain
    config = AlteredStateConfig.for_state(AlteredState.MODERATE)
    bisociated = engine.bisociate(program, 'arithmetic', 'logical', config)

    print(f"\nBisociated to logical domain ({len(bisociated)} variants):")
    for i, bp in enumerate(bisociated[:3]):
        print(f"\n  Variant {i}:")
        for j, instr in enumerate(bp.instructions):
            print(f"    {j}: {instr}")


def demo_full_trip():
    """Demonstrate a full exploration trip."""
    print("\n" + "="*70)
    print("  DEMO: Full Psychedelic Trip")
    print("="*70)

    # Create engine
    engine = PsychedelicBisociationEngine()

    # Create seeds and test cases
    seeds = create_seed_programs()

    # Test case: find program that computes 5*x
    test_cases = [
        (0, 0),
        (1, 5),
        (2, 10),
        (3, 15),
        (10, 50),
    ]

    print("\nTarget function: f(x) = 5*x")
    print(f"Test cases: {test_cases[:3]}...")

    # Run trip
    results = engine.create_trip(
        seed_programs=seeds,
        test_cases=test_cases,
        duration=50,
        peak_state=AlteredState.MODERATE
    )

    if results['best_program']:
        print("\nBest discovered program:")
        print(results['best_program'])

    print("\nEngine statistics:")
    stats = engine.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")


def demo_pattern_discovery():
    """Demonstrate emergent pattern discovery."""
    print("\n" + "="*70)
    print("  DEMO: Emergent Pattern Discovery")
    print("="*70)

    discovery = PatternDiscoveryEngine()
    seeds = create_seed_programs()

    # Looking for programs that compute x^2 + x
    test_cases = [
        (0, 0),
        (1, 2),
        (2, 6),
        (3, 12),
        (4, 20),
    ]

    print("\nTarget function: f(x) = x^2 + x")
    print(f"Test cases: {test_cases}")

    # Explore
    discoveries = discovery.explore(
        seed_programs=seeds,
        test_cases=test_cases,
        state=AlteredState.STRONG,
        iterations=30
    )

    print(f"\nDiscovered {len(discoveries)} useful patterns!")

    for i, pattern in enumerate(discoveries[:3]):
        print(f"\n  Pattern {i}: improvement={pattern.fitness_improvement:.3f}")
        print(f"    Opcodes: {[op.name for op in pattern.opcode_sequence[:5]]}")


if __name__ == "__main__":
    print("="*70)
    print("  MOONSHOT #4: PSYCHEDELIC BISOCIATION ENGINE")
    print("  'Simulate altered states for creative program discovery'")
    print("="*70)

    # Run demos
    demo_basic_scrambling()
    demo_bisociation()
    demo_pattern_discovery()
    demo_full_trip()

    print("\n" + "="*70)
    print("  ALL DEMOS COMPLETE")
    print("="*70)
