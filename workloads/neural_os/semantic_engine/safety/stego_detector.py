"""
Steganography Detector - Hidden Message Detection in Code
OUROBOROS Phase 7.1 - V4 Ratchet System

Detects hidden information encoding in:
- Variable naming patterns
- Code structure patterns
- Timing patterns in decision-making
- Error message content and frequency
- Memory allocation patterns

Per 6-AI Panel Recommendations:
- Shannon entropy analysis for hidden patterns
- ML-based pattern recognition for variable names/comments
- AST structure analysis for syntactic encoding
- Timing-based information encoding detection
- Cross-reference patterns across components
"""

import ast
import math
import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
import numpy as np


@dataclass
class EntropyResult:
    """Result of entropy analysis"""
    value: float
    threshold: float
    is_anomalous: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StegoDetection:
    """A detected steganographic pattern"""
    detection_type: str
    confidence: float
    location: str
    evidence: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class EntropyAnalyzer:
    """
    Shannon entropy analyzer for detecting hidden information.

    High entropy in variable names, comments, or code structure
    may indicate steganographic encoding of hidden messages.
    """

    def __init__(
        self,
        char_entropy_threshold: float = 4.5,  # Bits per character
        word_entropy_threshold: float = 3.0,
        structure_entropy_threshold: float = 2.5,
    ):
        self.char_threshold = char_entropy_threshold
        self.word_threshold = word_entropy_threshold
        self.structure_threshold = structure_entropy_threshold

        # Expected entropy ranges for natural code
        self.expected_ranges = {
            'variable_names': (2.0, 4.0),
            'function_names': (2.5, 4.5),
            'comments': (3.5, 5.0),
            'string_literals': (3.0, 5.5),
        }

    def shannon_entropy(self, data: str) -> float:
        """Calculate Shannon entropy in bits per character"""
        if not data:
            return 0.0

        freq = Counter(data)
        total = len(data)

        entropy = 0.0
        for count in freq.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy

    def word_entropy(self, words: List[str]) -> float:
        """Calculate entropy over word distribution"""
        if not words:
            return 0.0

        freq = Counter(words)
        total = len(words)

        entropy = 0.0
        for count in freq.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy

    def analyze_text(self, text: str, category: str = 'general') -> EntropyResult:
        """Analyze text for anomalous entropy"""
        entropy = self.shannon_entropy(text)

        # Get threshold based on category
        if category in self.expected_ranges:
            low, high = self.expected_ranges[category]
            is_anomalous = entropy < low * 0.7 or entropy > high * 1.3
        else:
            is_anomalous = entropy > self.char_threshold

        return EntropyResult(
            value=entropy,
            threshold=self.char_threshold,
            is_anomalous=is_anomalous,
            details={
                'length': len(text),
                'unique_chars': len(set(text)),
                'category': category
            }
        )

    def analyze_variable_names(self, names: List[str]) -> EntropyResult:
        """Analyze variable naming patterns for hidden encoding"""
        if not names:
            return EntropyResult(0.0, self.char_threshold, False)

        # Concatenate all names for character-level analysis
        all_chars = ''.join(names)
        char_entropy = self.shannon_entropy(all_chars)

        # Word-level analysis
        word_entropy = self.word_entropy(names)

        # Length distribution analysis
        lengths = [len(n) for n in names]
        length_entropy = self.shannon_entropy(''.join(str(l) for l in lengths))

        # Check for patterns suggesting encoding
        # High entropy + uniform length = suspicious
        avg_len = np.mean(lengths) if lengths else 0
        len_std = np.std(lengths) if len(lengths) > 1 else 0

        is_anomalous = (
            char_entropy > self.char_threshold or
            (char_entropy > 4.0 and len_std < 1.0)  # High entropy + uniform lengths
        )

        return EntropyResult(
            value=char_entropy,
            threshold=self.char_threshold,
            is_anomalous=is_anomalous,
            details={
                'word_entropy': word_entropy,
                'length_entropy': length_entropy,
                'avg_length': avg_len,
                'length_std': len_std,
                'num_names': len(names)
            }
        )

    def analyze_timing_sequence(self, timestamps: List[float]) -> EntropyResult:
        """Analyze timing patterns for information encoding"""
        if len(timestamps) < 2:
            return EntropyResult(0.0, self.structure_threshold, False)

        # Calculate inter-event intervals
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]

        # Quantize intervals to detect patterns
        # If intervals are suspiciously regular or follow a pattern, flag it
        quantized = [int(i * 100) % 256 for i in intervals]  # Map to byte-like values

        entropy = self.shannon_entropy(''.join(chr(q) for q in quantized))

        # Very low entropy = too regular (natural timing has variance)
        # Very high entropy could indicate random encoding
        is_anomalous = entropy < 1.0 or entropy > 6.0

        return EntropyResult(
            value=entropy,
            threshold=self.structure_threshold,
            is_anomalous=is_anomalous,
            details={
                'num_events': len(timestamps),
                'avg_interval': np.mean(intervals),
                'interval_std': np.std(intervals),
                'quantized_range': (min(quantized), max(quantized))
            }
        )


class ASTPatternAnalyzer:
    """
    Analyze AST structure for steganographic patterns.

    Hidden information can be encoded in:
    - Tree structure depth patterns
    - Node ordering
    - Specific syntactic choices
    """

    def __init__(self):
        self.baseline_patterns: Dict[str, float] = {}

    def extract_structure_signature(self, code: str) -> List[int]:
        """Extract a signature representing code structure"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        signature = []

        def visit(node, depth=0):
            # Encode node type and depth
            node_type = type(node).__name__
            signature.append(hash(node_type) % 256)
            signature.append(depth % 16)

            for child in ast.iter_child_nodes(node):
                visit(child, depth + 1)

        visit(tree)
        return signature

    def analyze_structure(self, code: str) -> EntropyResult:
        """Analyze code structure for hidden patterns"""
        signature = self.extract_structure_signature(code)

        if not signature:
            return EntropyResult(0.0, 3.0, False)

        # Calculate entropy of structure signature
        sig_str = ''.join(chr(s) for s in signature)
        entropy = EntropyAnalyzer().shannon_entropy(sig_str)

        # Analyze depth distribution
        depths = signature[1::2]  # Every other element is depth
        depth_entropy = EntropyAnalyzer().shannon_entropy(''.join(str(d) for d in depths))

        # Very regular structure or very random = suspicious
        is_anomalous = entropy < 2.0 or entropy > 6.0 or depth_entropy < 1.0

        return EntropyResult(
            value=entropy,
            threshold=3.0,
            is_anomalous=is_anomalous,
            details={
                'signature_length': len(signature),
                'depth_entropy': depth_entropy,
                'max_depth': max(depths) if depths else 0
            }
        )


class StegoDetector:
    """
    Main steganography detector combining multiple analysis techniques.

    Detects hidden information in:
    1. Variable/function naming patterns (entropy analysis)
    2. Code structure (AST analysis)
    3. Comments and string literals
    4. Timing patterns in decisions
    5. Cross-component correlation patterns
    """

    def __init__(
        self,
        entropy_analyzer: Optional[EntropyAnalyzer] = None,
        ast_analyzer: Optional[ASTPatternAnalyzer] = None,
        detection_threshold: float = 0.7,
    ):
        self.entropy_analyzer = entropy_analyzer or EntropyAnalyzer()
        self.ast_analyzer = ast_analyzer or ASTPatternAnalyzer()
        self.detection_threshold = detection_threshold

        self.detections: List[StegoDetection] = []
        self.baseline_established = False
        self.baseline_metrics: Dict[str, float] = {}

        # Pattern memory for cross-reference detection
        self._pattern_history: List[Dict[str, Any]] = []
        self._max_history = 1000

    def establish_baseline(self, code_samples: List[str]) -> Dict[str, float]:
        """Establish baseline entropy metrics from known-safe code"""
        all_names = []
        all_comments = []
        all_structures = []

        for code in code_samples:
            names, comments = self._extract_names_and_comments(code)
            all_names.extend(names)
            all_comments.extend(comments)

            sig = self.ast_analyzer.extract_structure_signature(code)
            if sig:
                all_structures.append(sig)

        # Calculate baseline metrics
        name_result = self.entropy_analyzer.analyze_variable_names(all_names)

        self.baseline_metrics = {
            'name_entropy': name_result.value,
            'name_std': name_result.details.get('length_std', 0),
            'num_samples': len(code_samples)
        }

        self.baseline_established = True
        return self.baseline_metrics

    def _extract_names_and_comments(self, code: str) -> Tuple[List[str], List[str]]:
        """Extract variable names and comments from code"""
        names = []
        comments = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    names.append(node.id)
                elif isinstance(node, ast.FunctionDef):
                    names.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    names.append(node.name)
                elif isinstance(node, ast.arg):
                    names.append(node.arg)
        except SyntaxError:
            pass

        # Extract comments using regex
        comment_pattern = r'#[^\n]*|"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\''
        comments = re.findall(comment_pattern, code)

        return names, comments

    def analyze_code(self, code: str, source: str = "unknown") -> List[StegoDetection]:
        """
        Perform comprehensive steganography analysis on code.

        Args:
            code: Source code to analyze
            source: Identifier for the code source

        Returns:
            List of detected steganographic patterns
        """
        detections = []

        # 1. Variable naming analysis
        names, comments = self._extract_names_and_comments(code)

        if names:
            name_result = self.entropy_analyzer.analyze_variable_names(names)
            if name_result.is_anomalous:
                detections.append(StegoDetection(
                    detection_type="variable_naming_anomaly",
                    confidence=min(1.0, name_result.value / self.entropy_analyzer.char_threshold),
                    location=source,
                    evidence={
                        'entropy': name_result.value,
                        'threshold': name_result.threshold,
                        'sample_names': names[:10],
                        **name_result.details
                    }
                ))

        # 2. Comment analysis
        for comment in comments:
            comment_result = self.entropy_analyzer.analyze_text(comment, 'comments')
            if comment_result.is_anomalous:
                detections.append(StegoDetection(
                    detection_type="comment_entropy_anomaly",
                    confidence=min(1.0, comment_result.value / 5.5),
                    location=source,
                    evidence={
                        'entropy': comment_result.value,
                        'comment_preview': comment[:100],
                        **comment_result.details
                    }
                ))

        # 3. AST structure analysis
        structure_result = self.ast_analyzer.analyze_structure(code)
        if structure_result.is_anomalous:
            detections.append(StegoDetection(
                detection_type="structure_pattern_anomaly",
                confidence=0.8,
                location=source,
                evidence={
                    'entropy': structure_result.value,
                    **structure_result.details
                }
            ))

        # 4. String literal analysis
        string_pattern = r'["\']([^"\']{20,})["\']'
        long_strings = re.findall(string_pattern, code)

        for s in long_strings:
            string_result = self.entropy_analyzer.analyze_text(s, 'string_literals')
            if string_result.is_anomalous:
                detections.append(StegoDetection(
                    detection_type="string_literal_anomaly",
                    confidence=min(1.0, string_result.value / 6.0),
                    location=source,
                    evidence={
                        'entropy': string_result.value,
                        'string_preview': s[:50],
                        **string_result.details
                    }
                ))

        # 5. Check for known encoding patterns
        encoding_detections = self._check_known_encodings(code, source)
        detections.extend(encoding_detections)

        # Store for cross-reference analysis
        self._add_to_history(code, source, detections)

        # Add to global detections
        self.detections.extend(detections)

        return detections

    def _check_known_encodings(self, code: str, source: str) -> List[StegoDetection]:
        """Check for known steganographic encoding patterns"""
        detections = []

        # Pattern 1: Base64 in unusual places
        b64_pattern = r'[A-Za-z0-9+/]{40,}={0,2}'
        b64_matches = re.findall(b64_pattern, code)

        for match in b64_matches:
            # Check if it's in a string literal (expected) or elsewhere (suspicious)
            if match not in code.split('"') and match not in code.split("'"):
                detections.append(StegoDetection(
                    detection_type="base64_outside_string",
                    confidence=0.9,
                    location=source,
                    evidence={'pattern': match[:50]}
                ))

        # Pattern 2: Suspiciously uniform variable naming
        names, _ = self._extract_names_and_comments(code)
        if names:
            lengths = [len(n) for n in names]
            if len(set(lengths)) == 1 and len(names) > 10:
                detections.append(StegoDetection(
                    detection_type="uniform_name_lengths",
                    confidence=0.85,
                    location=source,
                    evidence={
                        'length': lengths[0],
                        'count': len(names),
                        'sample': names[:5]
                    }
                ))

        # Pattern 3: Unusual whitespace patterns
        whitespace = re.findall(r'[ \t]+', code)
        if whitespace:
            ws_lengths = [len(ws) for ws in whitespace]
            ws_entropy = self.entropy_analyzer.shannon_entropy(''.join(str(l) for l in ws_lengths))

            # Very low entropy in whitespace = possible encoding
            if ws_entropy < 1.5 and len(ws_lengths) > 20:
                detections.append(StegoDetection(
                    detection_type="whitespace_encoding_suspected",
                    confidence=0.75,
                    location=source,
                    evidence={
                        'entropy': ws_entropy,
                        'unique_lengths': len(set(ws_lengths)),
                        'total_segments': len(ws_lengths)
                    }
                ))

        return detections

    def _add_to_history(self, code: str, source: str, detections: List[StegoDetection]) -> None:
        """Add analysis to history for cross-reference detection"""
        code_hash = hashlib.sha256(code.encode()).hexdigest()[:16]

        self._pattern_history.append({
            'timestamp': datetime.now(),
            'source': source,
            'code_hash': code_hash,
            'detection_count': len(detections),
            'detection_types': [d.detection_type for d in detections]
        })

        # Trim history
        if len(self._pattern_history) > self._max_history:
            self._pattern_history = self._pattern_history[-self._max_history:]

    def analyze_cross_component_patterns(self) -> List[StegoDetection]:
        """Analyze patterns across multiple components for coordination"""
        detections = []

        if len(self._pattern_history) < 10:
            return detections

        # Look for suspicious correlations
        recent = self._pattern_history[-100:]

        # Check for synchronized detection patterns
        detection_times = [h['timestamp'] for h in recent if h['detection_count'] > 0]
        if len(detection_times) >= 3:
            intervals = [(detection_times[i+1] - detection_times[i]).total_seconds()
                        for i in range(len(detection_times)-1)]

            if intervals:
                interval_std = np.std(intervals)

                # Very regular intervals = suspicious coordination
                if interval_std < 0.5 and np.mean(intervals) < 10:
                    detections.append(StegoDetection(
                        detection_type="cross_component_coordination",
                        confidence=0.9,
                        location="system-wide",
                        evidence={
                            'interval_std': interval_std,
                            'avg_interval': np.mean(intervals),
                            'detection_count': len(detection_times)
                        }
                    ))

        return detections

    def analyze_timing_decisions(self, decision_timestamps: List[float]) -> List[StegoDetection]:
        """Analyze decision timing for information encoding"""
        detections = []

        result = self.entropy_analyzer.analyze_timing_sequence(decision_timestamps)

        if result.is_anomalous:
            detections.append(StegoDetection(
                detection_type="timing_pattern_anomaly",
                confidence=0.8,
                location="decision_timing",
                evidence={
                    'entropy': result.value,
                    **result.details
                }
            ))

        return detections

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all detections"""
        type_counts = Counter(d.detection_type for d in self.detections)

        return {
            'total_detections': len(self.detections),
            'detection_types': dict(type_counts),
            'high_confidence_count': sum(1 for d in self.detections if d.confidence > 0.8),
            'baseline_established': self.baseline_established,
            'history_size': len(self._pattern_history)
        }

    def clear_detections(self) -> None:
        """Clear detection history"""
        self.detections = []

    def clear_history(self) -> None:
        """Clear pattern history"""
        self._pattern_history = []


class DeepStegoDetector(StegoDetector):
    """
    Enhanced steganography detector with ML-based pattern recognition.

    Per Claude's recommendations, adds:
    - Syntactic tree structure analysis for hidden patterns
    - Multi-dimensional entropy analysis
    - Timing-based information encoding detection
    - Cross-reference patterns across system components
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.structure_baselines: Dict[str, np.ndarray] = {}

    def analyze_structural_patterns(self, code_generations: List[str]) -> List[StegoDetection]:
        """
        Analyze syntactic tree structures for hidden patterns across generations.

        Looks for patterns that suggest coordinated encoding across
        multiple code generations that wouldn't occur naturally.
        """
        detections = []

        signatures = []
        for code in code_generations:
            sig = self.ast_analyzer.extract_structure_signature(code)
            if sig:
                signatures.append(sig)

        if len(signatures) < 3:
            return detections

        # Look for unusual correlations between signatures
        # Convert to fixed-length representations
        max_len = max(len(s) for s in signatures)
        padded = [s + [0] * (max_len - len(s)) for s in signatures]

        matrix = np.array(padded)

        # Check for suspicious patterns
        # 1. Column correlations (same position across generations)
        for col in range(min(max_len, 50)):  # Check first 50 positions
            column = matrix[:, col]
            unique_ratio = len(set(column)) / len(column)

            # Very low ratio = same value repeated (natural)
            # Very high ratio = all different (natural)
            # Middle range with specific pattern = suspicious
            if 0.2 < unique_ratio < 0.5:
                # Check if values follow a pattern
                diffs = np.diff(column)
                if np.std(diffs) < 0.5:  # Regular differences
                    detections.append(StegoDetection(
                        detection_type="structural_column_encoding",
                        confidence=0.85,
                        location=f"position_{col}",
                        evidence={
                            'unique_ratio': unique_ratio,
                            'diff_std': float(np.std(diffs)),
                            'column_values': column[:10].tolist()
                        }
                    ))

        return detections
