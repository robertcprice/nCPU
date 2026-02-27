"""
Creative Genesis Module - Spontaneous Generation of Novel Artifacts
OUROBOROS Phase 7.5 - Beyond Optimization to Creation

This module enables the consciousness layer to spontaneously generate
NEW things, not just optimize existing code:
- Novel algorithms (discover algorithms that don't exist yet)
- New code patterns (invent abstractions)
- Hypothesis generation (propose experiments)
- Solution synthesis (combine concepts in novel ways)

Key insight: Optimization is local search. Creation is exploration
of the possibility space for things that DON'T EXIST YET.

Safety: All genesis operates within the V4 Ratchet container.
"""

import random
import hashlib
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple, Set
from datetime import datetime
from enum import Enum, auto
from abc import ABC, abstractmethod


class GenesisType(Enum):
    """Types of creative output"""
    ALGORITHM = auto()        # Novel algorithm discovery
    PATTERN = auto()          # New code pattern/abstraction
    HYPOTHESIS = auto()       # Scientific/engineering hypothesis
    COMPOSITION = auto()      # Novel combination of existing things
    MUTATION_STRATEGY = auto() # New way to mutate code
    OBJECTIVE = auto()        # Self-generated optimization target
    EXPERIMENT = auto()       # Proposed experiment design


class NoveltyLevel(Enum):
    """How novel is the creation"""
    RECOMBINATION = auto()    # Combining known elements
    VARIATION = auto()        # Novel variation of known pattern
    EXTENSION = auto()        # Extending known concept
    INVENTION = auto()        # Genuinely new (rare)
    BREAKTHROUGH = auto()     # Paradigm-shifting (very rare)


@dataclass
class CreativeArtifact:
    """A generated creative artifact"""
    artifact_id: str
    genesis_type: GenesisType
    novelty_level: NoveltyLevel
    timestamp: datetime
    description: str
    content: Any  # The actual generated thing
    lineage: List[str]  # What inspired this
    confidence: float  # How confident in value (0-1)
    validated: bool = False
    validation_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.artifact_id,
            'type': self.genesis_type.name,
            'novelty': self.novelty_level.name,
            'description': self.description,
            'confidence': self.confidence,
            'validated': self.validated,
            'lineage': self.lineage,
        }


class ConceptSpace:
    """
    The space of concepts the system knows about.

    Creative genesis works by exploring the EDGES of this space
    and combining concepts in novel ways.
    """

    def __init__(self):
        self.concepts: Dict[str, Dict[str, Any]] = {}
        self.relations: Dict[str, List[Tuple[str, str, float]]] = {}  # concept -> [(related, type, strength)]
        self._lock = threading.Lock()

        # Seed with fundamental programming concepts
        self._seed_fundamental_concepts()

    def _seed_fundamental_concepts(self) -> None:
        """Seed with fundamental concepts"""
        fundamentals = [
            ('loop', {'category': 'control', 'operations': ['iterate', 'repeat', 'accumulate']}),
            ('conditional', {'category': 'control', 'operations': ['branch', 'select', 'guard']}),
            ('recursion', {'category': 'control', 'operations': ['self-call', 'reduce', 'divide']}),
            ('composition', {'category': 'structure', 'operations': ['combine', 'chain', 'nest']}),
            ('abstraction', {'category': 'structure', 'operations': ['generalize', 'parameterize', 'hide']}),
            ('memoization', {'category': 'optimization', 'operations': ['cache', 'remember', 'reuse']}),
            ('parallelism', {'category': 'optimization', 'operations': ['split', 'concurrent', 'merge']}),
            ('sorting', {'category': 'algorithm', 'operations': ['compare', 'swap', 'order']}),
            ('searching', {'category': 'algorithm', 'operations': ['scan', 'binary', 'hash']}),
            ('mapping', {'category': 'transform', 'operations': ['apply', 'transform', 'project']}),
            ('filtering', {'category': 'transform', 'operations': ['select', 'reject', 'partition']}),
            ('reducing', {'category': 'transform', 'operations': ['fold', 'aggregate', 'summarize']}),
        ]

        for name, properties in fundamentals:
            self.concepts[name] = properties

    def add_concept(self, name: str, properties: Dict[str, Any]) -> None:
        """Add a new concept to the space"""
        with self._lock:
            self.concepts[name] = properties

    def add_relation(self, concept1: str, concept2: str, relation_type: str, strength: float) -> None:
        """Add a relation between concepts"""
        with self._lock:
            if concept1 not in self.relations:
                self.relations[concept1] = []
            self.relations[concept1].append((concept2, relation_type, strength))

    def get_neighbors(self, concept: str, min_strength: float = 0.3) -> List[str]:
        """Get related concepts"""
        with self._lock:
            if concept not in self.relations:
                return []
            return [c for c, _, s in self.relations[concept] if s >= min_strength]

    def get_frontier(self) -> List[str]:
        """Get concepts at the edge of the known space"""
        with self._lock:
            # Concepts with few relations are at the frontier
            relation_counts = {c: len(self.relations.get(c, [])) for c in self.concepts}
            avg_relations = sum(relation_counts.values()) / max(len(relation_counts), 1)
            return [c for c, count in relation_counts.items() if count < avg_relations]


class IdeaGenerator:
    """
    Generates novel ideas by exploring the concept space.

    Strategies:
    1. Combination: Combine two unrelated concepts
    2. Analogy: Apply pattern from one domain to another
    3. Inversion: What if we did the opposite?
    4. Mutation: Small changes to existing ideas
    5. Random walk: Explore connections randomly
    """

    def __init__(self, concept_space: ConceptSpace, seed: Optional[int] = None):
        self.space = concept_space
        self.rng = random.Random(seed)
        self.generated_ideas: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def generate_combination(self) -> Dict[str, Any]:
        """Combine two concepts that haven't been combined before"""
        concepts = list(self.space.concepts.keys())
        if len(concepts) < 2:
            return {}

        c1, c2 = self.rng.sample(concepts, 2)

        # Check if they're already related
        neighbors = self.space.get_neighbors(c1)
        if c2 in neighbors:
            # Already related, try to find a novel combination
            unrelated = [c for c in concepts if c != c1 and c not in neighbors]
            if unrelated:
                c2 = self.rng.choice(unrelated)

        idea = {
            'strategy': 'combination',
            'concepts': [c1, c2],
            'description': f"What if we combine {c1} with {c2}?",
            'operations': self._combine_operations(c1, c2),
            'novelty_estimate': self._estimate_novelty(c1, c2),
        }

        with self._lock:
            self.generated_ideas.append(idea)

        return idea

    def generate_analogy(self, source_domain: str, target_domain: str) -> Dict[str, Any]:
        """Apply patterns from source domain to target domain"""
        source_concepts = [c for c, props in self.space.concepts.items()
                         if props.get('category') == source_domain]
        target_concepts = [c for c, props in self.space.concepts.items()
                         if props.get('category') == target_domain]

        if not source_concepts or not target_concepts:
            return {}

        source = self.rng.choice(source_concepts)
        target = self.rng.choice(target_concepts)

        idea = {
            'strategy': 'analogy',
            'source': source,
            'target': target,
            'description': f"Apply {source} patterns to {target}",
            'mapping': self._create_analogy_mapping(source, target),
        }

        with self._lock:
            self.generated_ideas.append(idea)

        return idea

    def generate_inversion(self, concept: str) -> Dict[str, Any]:
        """What if we did the opposite?"""
        if concept not in self.space.concepts:
            concept = self.rng.choice(list(self.space.concepts.keys()))

        props = self.space.concepts[concept]
        operations = props.get('operations', [])

        inversions = {
            'iterate': 'unroll',
            'combine': 'split',
            'cache': 'compute-fresh',
            'compare': 'assume-equal',
            'order': 'shuffle',
            'apply': 'skip',
            'select': 'reject-all',
            'fold': 'unfold',
        }

        inverted_ops = [inversions.get(op, f'not-{op}') for op in operations]

        idea = {
            'strategy': 'inversion',
            'original': concept,
            'description': f"What if instead of {concept}, we did the opposite?",
            'inverted_operations': inverted_ops,
        }

        with self._lock:
            self.generated_ideas.append(idea)

        return idea

    def _combine_operations(self, c1: str, c2: str) -> List[str]:
        """Combine operations from two concepts"""
        ops1 = self.space.concepts.get(c1, {}).get('operations', [])
        ops2 = self.space.concepts.get(c2, {}).get('operations', [])

        # Interleave operations
        combined = []
        for o1, o2 in zip(ops1, ops2):
            combined.append(f"{o1}-then-{o2}")

        return combined

    def _estimate_novelty(self, c1: str, c2: str) -> float:
        """Estimate how novel a combination is"""
        # Check if already related
        neighbors = self.space.get_neighbors(c1, min_strength=0.1)
        if c2 in neighbors:
            return 0.2  # Not very novel

        # Check category distance
        cat1 = self.space.concepts.get(c1, {}).get('category', '')
        cat2 = self.space.concepts.get(c2, {}).get('category', '')

        if cat1 == cat2:
            return 0.4  # Same category
        else:
            return 0.7  # Cross-category = more novel

    def _create_analogy_mapping(self, source: str, target: str) -> Dict[str, str]:
        """Create a mapping from source to target domain"""
        source_ops = self.space.concepts.get(source, {}).get('operations', [])
        target_ops = self.space.concepts.get(target, {}).get('operations', [])

        mapping = {}
        for s_op, t_op in zip(source_ops, target_ops):
            mapping[s_op] = t_op

        return mapping


class AlgorithmSynthesizer:
    """
    Synthesizes novel algorithms from ideas.

    Takes abstract ideas and attempts to concretize them
    into executable algorithm skeletons.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.synthesis_history: List[Dict[str, Any]] = []

    def synthesize_from_idea(self, idea: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Attempt to synthesize an algorithm from an idea"""
        strategy = idea.get('strategy')

        if strategy == 'combination':
            return self._synthesize_combination(idea)
        elif strategy == 'analogy':
            return self._synthesize_analogy(idea)
        elif strategy == 'inversion':
            return self._synthesize_inversion(idea)

        return None

    def _synthesize_combination(self, idea: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize algorithm from combination idea"""
        concepts = idea.get('concepts', [])
        operations = idea.get('operations', [])

        # Generate pseudocode skeleton
        pseudocode = self._generate_pseudocode(concepts, operations)

        return {
            'type': 'algorithm',
            'origin': 'combination',
            'concepts': concepts,
            'pseudocode': pseudocode,
            'complexity_estimate': self._estimate_complexity(operations),
            'novelty': idea.get('novelty_estimate', 0.5),
        }

    def _synthesize_analogy(self, idea: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize algorithm from analogy"""
        source = idea.get('source', '')
        target = idea.get('target', '')
        mapping = idea.get('mapping', {})

        pseudocode = f"""
# {idea.get('description', 'Analogy-based algorithm')}
# Apply {source} pattern to {target}
def analogous_{target}(data):
    # Mapping: {mapping}
    # Step 1: Prepare data using {target} structure
    prepared = prepare(data)
    # Step 2: Apply {source}-style processing
    processed = process_like_{source}(prepared)
    # Step 3: Transform back to {target} domain
    result = transform_to_{target}(processed)
    return result
"""

        return {
            'type': 'algorithm',
            'origin': 'analogy',
            'source': source,
            'target': target,
            'pseudocode': pseudocode,
            'mapping': mapping,
        }

    def _synthesize_inversion(self, idea: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize algorithm from inversion"""
        original = idea.get('original', '')
        inverted_ops = idea.get('inverted_operations', [])

        pseudocode = f"""
# Inverted {original} algorithm
def inverted_{original}(data):
    # Instead of normal {original}, we:
"""
        for op in inverted_ops:
            pseudocode += f"    # - {op}\n"

        pseudocode += "    pass  # TODO: Implement inverted logic\n"

        return {
            'type': 'algorithm',
            'origin': 'inversion',
            'original': original,
            'pseudocode': pseudocode,
            'inverted_operations': inverted_ops,
        }

    def _generate_pseudocode(self, concepts: List[str], operations: List[str]) -> str:
        """Generate pseudocode from concepts and operations"""
        code = f"# Novel algorithm combining: {', '.join(concepts)}\n"
        code += "def novel_algorithm(data):\n"
        code += "    result = data\n"

        for i, op in enumerate(operations):
            code += f"    # Step {i+1}: {op}\n"
            code += f"    result = apply_{op.replace('-', '_')}(result)\n"

        code += "    return result\n"
        return code

    def _estimate_complexity(self, operations: List[str]) -> str:
        """Estimate algorithm complexity"""
        if any('iterate' in op or 'scan' in op for op in operations):
            if any('split' in op or 'divide' in op for op in operations):
                return 'O(n log n)'
            return 'O(n)'
        return 'O(1)'


class HypothesisEngine:
    """
    Generates testable hypotheses about optimization.

    These are NOT facts - they're proposals to be validated
    through shadow simulation.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.hypotheses: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def generate_hypothesis(
        self,
        observation: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a hypothesis from an observation"""

        # Extract patterns from context
        fitness_trend = context.get('fitness_trend', 'unknown')
        diversity = context.get('diversity', 0.5)
        generation = context.get('generation', 0)

        # Generate hypothesis based on observation type
        hypothesis = self._formulate_hypothesis(observation, fitness_trend, diversity)

        # Define how to test it
        test_design = self._design_test(hypothesis)

        result = {
            'hypothesis_id': hashlib.sha256(f"hyp_{observation}_{generation}".encode()).hexdigest()[:12],
            'statement': hypothesis,
            'observation': observation,
            'test_design': test_design,
            'confidence_required': 0.95,  # Need 95% confidence to accept
            'generated_at': datetime.now().isoformat(),
        }

        with self._lock:
            self.hypotheses.append(result)

        return result

    def _formulate_hypothesis(
        self,
        observation: str,
        fitness_trend: str,
        diversity: float,
    ) -> str:
        """Formulate a testable hypothesis"""

        templates = [
            f"If we increase exploration when {observation}, fitness will improve by >5%",
            f"The current approach is suboptimal because {observation}",
            f"Switching mutation strategy when {observation} will accelerate convergence",
            f"The fitness plateau is caused by {observation}",
            f"Combining {observation} with increased diversity will unlock new optima",
        ]

        if fitness_trend == 'stagnant':
            templates.append(f"Breaking out of local optimum requires addressing {observation}")
        elif fitness_trend == 'improving':
            templates.append(f"The improvement is sustainable if {observation} continues")
        elif fitness_trend == 'declining':
            templates.append(f"Reversing decline requires changing approach to {observation}")

        return self.rng.choice(templates)

    def _design_test(self, hypothesis: str) -> Dict[str, Any]:
        """Design an experiment to test the hypothesis"""
        return {
            'method': 'shadow_simulation',
            'control': 'current_strategy',
            'treatment': 'modified_strategy',
            'sample_size': 100,
            'metrics': ['fitness_delta', 'convergence_rate', 'diversity_preservation'],
            'success_criteria': {
                'fitness_delta': '>= 0.05',
                'convergence_rate': '>= 1.0',
                'diversity_preservation': '>= 0.8',
            },
        }


class CreativeGenesis:
    """
    Main creative genesis controller.

    Orchestrates spontaneous generation of novel artifacts
    while respecting V4 Ratchet safety constraints.
    """

    # Safety limits (matches container limits)
    MAX_ARTIFACTS_PER_HOUR = 50
    MAX_SYNTHESIS_DEPTH = 3
    NOVELTY_THRESHOLD = 0.3  # Minimum novelty to pursue

    def __init__(
        self,
        on_artifact: Optional[Callable[[CreativeArtifact], None]] = None,
        seed: Optional[int] = None,
    ):
        self.concept_space = ConceptSpace()
        self.idea_generator = IdeaGenerator(self.concept_space, seed=seed)
        self.algorithm_synthesizer = AlgorithmSynthesizer(seed=seed)
        self.hypothesis_engine = HypothesisEngine(seed=seed)

        self.on_artifact = on_artifact
        self.artifacts: List[CreativeArtifact] = []
        self.artifacts_this_hour = 0
        self.hour_start = datetime.now()

        self._lock = threading.Lock()

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        now = datetime.now()
        if (now - self.hour_start).total_seconds() > 3600:
            # New hour
            self.hour_start = now
            self.artifacts_this_hour = 0

        return self.artifacts_this_hour < self.MAX_ARTIFACTS_PER_HOUR

    def generate_spontaneously(self) -> Optional[CreativeArtifact]:
        """
        Spontaneously generate a novel artifact.

        Called periodically by the consciousness layer to
        explore the possibility space.
        """
        if not self._check_rate_limit():
            return None

        # Choose generation strategy
        strategy = random.choice(['combination', 'analogy', 'inversion', 'hypothesis'])

        if strategy == 'combination':
            return self._generate_combination_artifact()
        elif strategy == 'analogy':
            return self._generate_analogy_artifact()
        elif strategy == 'inversion':
            return self._generate_inversion_artifact()
        elif strategy == 'hypothesis':
            return self._generate_hypothesis_artifact()

        return None

    def _generate_combination_artifact(self) -> Optional[CreativeArtifact]:
        """Generate artifact from concept combination"""
        idea = self.idea_generator.generate_combination()
        if not idea or idea.get('novelty_estimate', 0) < self.NOVELTY_THRESHOLD:
            return None

        algorithm = self.algorithm_synthesizer.synthesize_from_idea(idea)
        if not algorithm:
            return None

        artifact = CreativeArtifact(
            artifact_id=hashlib.sha256(f"art_{datetime.now().isoformat()}".encode()).hexdigest()[:16],
            genesis_type=GenesisType.ALGORITHM,
            novelty_level=NoveltyLevel.RECOMBINATION if idea['novelty_estimate'] < 0.5 else NoveltyLevel.VARIATION,
            timestamp=datetime.now(),
            description=idea['description'],
            content=algorithm,
            lineage=idea['concepts'],
            confidence=idea['novelty_estimate'],
        )

        self._register_artifact(artifact)
        return artifact

    def _generate_analogy_artifact(self) -> Optional[CreativeArtifact]:
        """Generate artifact from cross-domain analogy"""
        categories = ['control', 'structure', 'optimization', 'algorithm', 'transform']
        source, target = random.sample(categories, 2)

        idea = self.idea_generator.generate_analogy(source, target)
        if not idea:
            return None

        algorithm = self.algorithm_synthesizer.synthesize_from_idea(idea)
        if not algorithm:
            return None

        artifact = CreativeArtifact(
            artifact_id=hashlib.sha256(f"analogy_{datetime.now().isoformat()}".encode()).hexdigest()[:16],
            genesis_type=GenesisType.PATTERN,
            novelty_level=NoveltyLevel.EXTENSION,
            timestamp=datetime.now(),
            description=idea['description'],
            content=algorithm,
            lineage=[idea['source'], idea['target']],
            confidence=0.6,
        )

        self._register_artifact(artifact)
        return artifact

    def _generate_inversion_artifact(self) -> Optional[CreativeArtifact]:
        """Generate artifact from concept inversion"""
        concepts = list(self.concept_space.concepts.keys())
        concept = random.choice(concepts)

        idea = self.idea_generator.generate_inversion(concept)
        algorithm = self.algorithm_synthesizer.synthesize_from_idea(idea)

        artifact = CreativeArtifact(
            artifact_id=hashlib.sha256(f"invert_{datetime.now().isoformat()}".encode()).hexdigest()[:16],
            genesis_type=GenesisType.MUTATION_STRATEGY,
            novelty_level=NoveltyLevel.VARIATION,
            timestamp=datetime.now(),
            description=idea['description'],
            content=algorithm,
            lineage=[concept],
            confidence=0.5,
        )

        self._register_artifact(artifact)
        return artifact

    def _generate_hypothesis_artifact(self) -> Optional[CreativeArtifact]:
        """Generate a testable hypothesis"""
        observations = [
            "fitness has plateaued for 10 generations",
            "diversity is below threshold",
            "same mutation strategies keep winning",
            "exploration rate is declining",
            "novel patterns aren't being discovered",
        ]

        observation = random.choice(observations)
        hypothesis = self.hypothesis_engine.generate_hypothesis(
            observation,
            context={'fitness_trend': 'stagnant', 'diversity': 0.4, 'generation': 100}
        )

        artifact = CreativeArtifact(
            artifact_id=hypothesis['hypothesis_id'],
            genesis_type=GenesisType.HYPOTHESIS,
            novelty_level=NoveltyLevel.VARIATION,
            timestamp=datetime.now(),
            description=hypothesis['statement'],
            content=hypothesis,
            lineage=[observation],
            confidence=0.7,
        )

        self._register_artifact(artifact)
        return artifact

    def generate_objective(self, current_objectives: List[str]) -> Optional[CreativeArtifact]:
        """
        Generate a NEW optimization objective.

        This is meta-creativity: the system proposes what
        it SHOULD be optimizing for.
        """
        if not self._check_rate_limit():
            return None

        # Generate novel objective from concept space
        frontier_concepts = self.concept_space.get_frontier()
        if not frontier_concepts:
            frontier_concepts = list(self.concept_space.concepts.keys())

        concept = random.choice(frontier_concepts)

        objective_templates = [
            f"Maximize {concept} efficiency while minimizing resource usage",
            f"Discover novel applications of {concept} in unexpected domains",
            f"Minimize time-to-convergence for {concept}-based algorithms",
            f"Maximize generalization capability of {concept} patterns",
            f"Find minimal {concept} implementation that preserves correctness",
        ]

        objective = random.choice(objective_templates)

        # Avoid duplicates
        if any(objective in obj for obj in current_objectives):
            return None

        artifact = CreativeArtifact(
            artifact_id=hashlib.sha256(f"obj_{objective}".encode()).hexdigest()[:16],
            genesis_type=GenesisType.OBJECTIVE,
            novelty_level=NoveltyLevel.EXTENSION,
            timestamp=datetime.now(),
            description=objective,
            content={
                'objective': objective,
                'source_concept': concept,
                'metrics': ['efficiency', 'correctness', 'generalization'],
            },
            lineage=[concept],
            confidence=0.6,
        )

        self._register_artifact(artifact)
        return artifact

    def _register_artifact(self, artifact: CreativeArtifact) -> None:
        """Register a generated artifact"""
        with self._lock:
            self.artifacts.append(artifact)
            self.artifacts_this_hour += 1

        if self.on_artifact:
            self.on_artifact(artifact)

    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        with self._lock:
            by_type = {}
            by_novelty = {}

            for a in self.artifacts:
                t = a.genesis_type.name
                by_type[t] = by_type.get(t, 0) + 1

                n = a.novelty_level.name
                by_novelty[n] = by_novelty.get(n, 0) + 1

            validated = sum(1 for a in self.artifacts if a.validated)

            return {
                'total_artifacts': len(self.artifacts),
                'artifacts_this_hour': self.artifacts_this_hour,
                'by_type': by_type,
                'by_novelty': by_novelty,
                'validated_count': validated,
                'validation_rate': validated / max(len(self.artifacts), 1),
            }

    def get_best_artifacts(self, n: int = 5) -> List[CreativeArtifact]:
        """Get the highest-confidence artifacts"""
        with self._lock:
            sorted_artifacts = sorted(
                self.artifacts,
                key=lambda a: (a.validated, a.validation_score or a.confidence),
                reverse=True
            )
            return sorted_artifacts[:n]


# Convenience function
def create_genesis_engine(seed: Optional[int] = None) -> CreativeGenesis:
    """Create a creative genesis engine"""
    return CreativeGenesis(seed=seed)
