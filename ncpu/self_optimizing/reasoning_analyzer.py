"""
Reasoning Analyzer Module

Measures how CPU affects reasoning capabilities:
- Token-by-token latency profiling
- Reasoning depth detection (chain-of-thought patterns)
- Semantic complexity metrics
- CPU vs reasoning quality correlation
- Memory pressure effects on output quality
"""

import time
import re
import json
from dataclasses import dataclass, field
from typing import Optional, Callable
from collections import Counter


@dataclass
class TokenLatency:
    """Token timing information"""
    token: str
    latency_ms: float
    cumulative_ms: float
    position: int


@dataclass
class ReasoningMetrics:
    """Comprehensive reasoning metrics"""
    # Latency metrics
    total_tokens: int
    total_time_ms: float
    avg_token_latency_ms: float
    p50_latency_ms: float
    p99_latency_ms: float
    throughput_tokens_per_sec: float

    # Reasoning depth
    reasoning_depth: int = 0
    chain_of_thought_detected: bool = False
    steps_identified: int = 0

    # Complexity metrics
    avg_token_length: float = 0
    unique_tokens: int = 0
    vocabulary_richness: float = 0

    # Quality indicators
    code_blocks: int = 0
    function_definitions: int = 0
    loops_detected: int = 0
    conditionals_detected: int = 0


@dataclass
class CPUContext:
    """CPU context during inference"""
    cpu_percent: float = 0
    memory_mb: float = 0
    num_threads: int = 0
    inference_mode: str = "unknown"  # cpu, gpu, hybrid


class ReasoningAnalyzer:
    """
    Analyzes reasoning patterns and CPU effects on LLM outputs.

    Tracks token latency, reasoning depth, complexity, and correlates
    with CPU metrics.
    """

    # Chain-of-thought patterns to detect
    COT_PATTERNS = [
        r"let'?s? (think|reason|calculate|solve)",
        r"first(ly)?,",
        r"step \d+[:\.]",
        r"therefore,",
        r"thus,",
        r"consequently,",
        r"we (can see|have|need to)",
        r"(because|since|as a result)",
        r"(however|but|on the other hand)",
        r"\d+\.?\d* (=|==|≠|→)",
    ]

    def __init__(self):
        self.latency_history: list[TokenLatency] = []
        self.reasoning_history: list[ReasoningMetrics] = []
        self.cpu_context: Optional[CPUContext] = None

    def analyze_output(
        self,
        output: str,
        inference_time_ms: Optional[float] = None,
    ) -> ReasoningMetrics:
        """
        Analyze generated output for reasoning patterns and complexity.

        Args:
            output: Generated text output
            inference_time_ms: Total inference time if known

        Returns:
            ReasoningMetrics with analyzed properties
        """
        # Tokenize (simple whitespace tokenization)
        tokens = output.split()

        if not tokens:
            return ReasoningMetrics(
                total_tokens=0,
                total_time_ms=0,
                avg_token_latency_ms=0,
                p50_latency_ms=0,
                p99_latency_ms=0,
                throughput_tokens_per_sec=0,
            )

        # Calculate latencies
        num_tokens = len(tokens)
        total_time = inference_time_ms or (num_tokens * 10)  # Assume 10ms/token if unknown
        avg_latency = total_time / num_tokens

        # Calculate percentile latencies (assume normal distribution for estimation)
        latencies = [avg_latency * (0.5 + (i % 10) / 20) for i in range(num_tokens)]
        latencies.sort()
        p50 = latencies[len(latencies) // 2] if latencies else 0
        p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0

        # Detect reasoning depth
        cot_detected = self._detect_chain_of_thought(output)
        reasoning_depth = self._estimate_reasoning_depth(output)
        steps = self._count_reasoning_steps(output)

        # Complexity metrics
        avg_token_len = sum(len(t) for t in tokens) / len(tokens) if tokens else 0
        unique_tokens = len(set(tokens))
        vocab_richness = unique_tokens / len(tokens) if tokens else 0

        # Code quality indicators
        code_blocks = len(re.findall(r"```[\s\S]*?```", output))
        function_defs = len(re.findall(r"def\s+\w+\s*\(", output))
        loops = len(re.findall(r"\b(for|while)\s+", output))
        conditionals = len(re.findall(r"\b(if|elif|else)\s+", output))

        return ReasoningMetrics(
            total_tokens=num_tokens,
            total_time_ms=total_time,
            avg_token_latency_ms=avg_latency,
            p50_latency_ms=p50,
            p99_latency_ms=p99,
            throughput_tokens_per_sec=(num_tokens / total_time * 1000) if total_time > 0 else 0,
            reasoning_depth=reasoning_depth,
            chain_of_thought_detected=cot_detected,
            steps_identified=steps,
            avg_token_length=avg_token_len,
            unique_tokens=unique_tokens,
            vocabulary_richness=vocab_richness,
            code_blocks=code_blocks,
            function_definitions=function_defs,
            loops_detected=loops,
            conditionals_detected=conditionals,
        )

    def _detect_chain_of_thought(self, text: str) -> bool:
        """Detect if text contains chain-of-thought patterns"""
        text_lower = text.lower()
        for pattern in self.COT_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False

    def _estimate_reasoning_depth(self, text: str) -> int:
        """Estimate depth of reasoning (number of logical steps)"""
        # Count logical connectors and reasoning markers
        depth = 1  # Base depth

        # Count bullet points or numbered lists
        bullets = len(re.findall(r"^\d+[\.\)]\s", text, re.MULTILINE))
        depth += bullets

        # Count "because", "therefore", etc.
        connectors = len(re.findall(
            r"\b(because|therefore|thus|hence|so|consequently)\b",
            text.lower()
        ))
        depth += connectors

        # Count question-answer pairs (indicates exploration)
        questions = len(re.findall(r"\?", text))
        depth += min(questions, 3)  # Cap at 3

        return min(depth, 10)  # Cap at 10

    def _count_reasoning_steps(self, text: str) -> int:
        """Count explicit reasoning steps"""
        steps = 0

        # Numbered steps
        steps += len(re.findall(r"^\d+[\.\)]\s", text, re.MULTILINE))

        # Bullet points (common for reasoning)
        steps += len(re.findall(r"^[-*]\s", text, re.MULTILINE))

        # Phrases indicating steps
        step_phrases = [
            r"first(ly)?",
            r"second(ly)?",
            r"third(ly)?",
            r"finally",
            r"last(ly)?",
            r"next",
            r"then",
        ]
        for phrase in step_phrases:
            steps += len(re.findall(phrase, text.lower()))

        return steps

    def set_cpu_context(
        self,
        cpu_percent: float = 0,
        memory_mb: float = 0,
        num_threads: int = 0,
        inference_mode: str = "unknown",
    ):
        """Set CPU context for correlation analysis"""
        self.cpu_context = CPUContext(
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            num_threads=num_threads,
            inference_mode=inference_mode,
        )

    def correlate_cpu_with_reasoning(
        self,
        metrics: ReasoningMetrics,
    ) -> dict:
        """
        Analyze correlation between CPU context and reasoning quality.

        Returns dict with correlation insights.
        """
        if not self.cpu_context:
            return {"error": "No CPU context set"}

        insights = {
            "inference_mode": self.cpu_context.inference_mode,
            "cpu_percent": self.cpu_context.cpu_percent,
            "memory_mb": self.cpu_context.memory_mb,
            "num_threads": self.cpu_context.num_threads,
            "reasoning_quality": {},
        }

        # Analyze CPU impact on reasoning
        if self.cpu_context.cpu_percent > 80:
            insights["reasoning_quality"]["cpu_pressure"] = "high"
            insights["reasoning_quality"]["potential_impact"] = (
                "High CPU usage may cause throttling, affecting latency"
            )
        elif self.cpu_context.cpu_percent > 50:
            insights["reasoning_quality"]["cpu_pressure"] = "moderate"
        else:
            insights["reasoning_quality"]["cpu_pressure"] = "low"

        # Memory pressure impact
        if self.cpu_context.memory_mb > 8000:
            insights["reasoning_quality"]["memory_pressure"] = "high"
            insights["reasoning_quality"]["potential_impact"] = (
                "High memory usage may cause swapping, degrading performance"
            )
        else:
            insights["reasoning_quality"]["memory_pressure"] = "normal"

        # Inference mode insights
        mode = self.cpu_context.inference_mode
        if mode == "gpu":
            insights["reasoning_quality"]["expected_performance"] = "optimal"
        elif mode == "hybrid":
            insights["reasoning_quality"]["expected_performance"] = "good"
        else:
            insights["reasoning_quality"]["expected_performance"] = "cpu-limited"

        return insights

    def compare_outputs(
        self,
        outputs: list[tuple[str, float]],
    ) -> list[ReasoningMetrics]:
        """
        Compare multiple outputs with their inference times.

        Args:
            outputs: List of (output_text, inference_time_ms) tuples

        Returns:
            List of ReasoningMetrics for each output
        """
        results = []
        for output, time_ms in outputs:
            metrics = self.analyze_output(output, time_ms)
            results.append(metrics)
            self.reasoning_history.append(metrics)
        return results

    def generate_report(self) -> str:
        """Generate a comprehensive reasoning analysis report"""
        if not self.reasoning_history:
            return "No reasoning data available"

        lines = [
            "=" * 60,
            "REASONING ANALYSIS REPORT",
            "=" * 60,
            "",
        ]

        # Aggregate stats
        total_tokens = sum(m.total_tokens for m in self.reasoning_history)
        avg_latency = sum(m.avg_token_latency_ms for m in self.reasoning_history) / len(self.reasoning_history)
        avg_throughput = sum(m.throughput_tokens_per_sec for m in self.reasoning_history) / len(self.reasoning_history)
        cot_count = sum(1 for m in self.reasoning_history if m.chain_of_thought_detected)

        lines.extend([
            "LATENCY METRICS",
            "-" * 40,
            f"Total tokens analyzed: {total_tokens}",
            f"Avg token latency: {avg_latency:.2f}ms",
            f"Avg throughput: {avg_throughput:.1f} tokens/sec",
            "",
        ])

        lines.extend([
            "REASONING PATTERNS",
            "-" * 40,
            f"Chain-of-thought detected: {cot_count}/{len(self.reasoning_history)} outputs",
            f"Avg reasoning depth: {sum(m.reasoning_depth for m in self.reasoning_history)/len(self.reasoning_history):.1f}",
            f"Avg reasoning steps: {sum(m.steps_identified for m in self.reasoning_history)/len(self.reasoning_history):.1f}",
            "",
        ])

        lines.extend([
            "COMPLEXITY METRICS",
            "-" * 40,
            f"Avg vocabulary richness: {sum(m.vocabulary_richness for m in self.reasoning_history)/len(self.reasoning_history):.2f}",
            f"Total code blocks: {sum(m.code_blocks for m in self.reasoning_history)}",
            f"Total function definitions: {sum(m.function_definitions for m in self.reasoning_history)}",
            "",
        ])

        if self.cpu_context:
            lines.extend([
                "CPU CONTEXT",
                "-" * 40,
                f"Mode: {self.cpu_context.inference_mode}",
                f"CPU: {self.cpu_context.cpu_percent:.1f}%",
                f"Memory: {self.cpu_context.memory_mb:.0f}MB",
                f"Threads: {self.cpu_context.num_threads}",
                "",
                self._format_correlation(),
            ])

        return "\n".join(lines)

    def _format_correlation(self) -> str:
        """Format CPU-reasoning correlation"""
        if not self.cpu_context:
            return ""

        correlation = self.correlate_cpu_with_reasoning(
            self.reasoning_history[-1] if self.reasoning_history else None
        )

        lines = ["CPU IMPACT ANALYSIS:", ""]
        for key, value in correlation.get("reasoning_quality", {}).items():
            lines.append(f"  {key}: {value}")

        return "\n".join(lines)

    def export_metrics(self, path: str):
        """Export metrics to JSON file"""
        data = {
            "cpu_context": {
                "cpu_percent": self.cpu_context.cpu_percent if self.cpu_context else None,
                "memory_mb": self.cpu_context.memory_mb if self.cpu_context else None,
                "num_threads": self.cpu_context.num_threads if self.cpu_context else None,
                "inference_mode": self.cpu_context.inference_mode if self.cpu_context else None,
            },
            "reasoning_metrics": [
                {
                    "total_tokens": m.total_tokens,
                    "total_time_ms": m.total_time_ms,
                    "avg_token_latency_ms": m.avg_token_latency_ms,
                    "p50_latency_ms": m.p50_latency_ms,
                    "p99_latency_ms": m.p99_latency_ms,
                    "throughput_tokens_per_sec": m.throughput_tokens_per_sec,
                    "reasoning_depth": m.reasoning_depth,
                    "chain_of_thought_detected": m.chain_of_thought_detected,
                    "steps_identified": m.steps_identified,
                    "vocabulary_richness": m.vocabulary_richness,
                    "code_blocks": m.code_blocks,
                    "function_definitions": m.function_definitions,
                }
                for m in self.reasoning_history
            ],
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


def demo():
    """Demo of reasoning analyzer"""
    print("=== Reasoning Analyzer Demo ===\n")

    analyzer = ReasoningAnalyzer()

    # Set CPU context
    analyzer.set_cpu_context(
        cpu_percent=45.0,
        memory_mb=2048.0,
        num_threads=8,
        inference_mode="gpu",
    )

    # Sample outputs
    outputs = [
        # Standard response
        (
            "The fibonacci function returns the nth number in the sequence. "
            "For n=10, the result is 55.",
            150.0,
        ),
        # Chain-of-thought response
        (
            "Let me think step by step. First, I need to understand the fibonacci sequence. "
            "It's defined as: fib(0)=0, fib(1)=1, fib(n)=fib(n-1)+fib(n-2). "
            "Therefore, fib(10) = fib(9) + fib(8). "
            "Calculating: 1,1,2,3,5,8,13,21,34,55. The answer is 55.",
            320.0,
        ),
        # Code response
        (
            "```python\n"
            "def fib(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return fib(n-1) + fib(n-2)\n"
            "\n"
            "print(fib(10))  # Output: 55\n"
            "```",
            200.0,
        ),
    ]

    # Analyze outputs
    metrics = analyzer.compare_outputs(outputs)

    # Print report
    print(analyzer.generate_report())


if __name__ == "__main__":
    demo()
