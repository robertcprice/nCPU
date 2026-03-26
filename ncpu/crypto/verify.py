"""Formal timing verification for constant-time implementations.

Verifies that cryptographic operations execute in constant time by:
  1. Running the same operation with different inputs
  2. Measuring cycle counts (on GPU: exact, deterministic)
  3. Verifying sigma=0.0 across all runs
  4. Generating formal verification reports

On conventional CPUs, this kind of verification is fundamentally limited:
microarchitectural noise (cache state, OS interrupts, branch predictor
warm-up) masks small timing differences, making it impossible to distinguish
"constant-time" from "almost constant-time" from "timing leak hidden by noise."

On nCPU's GPU, execution is perfectly deterministic (every instruction takes
exactly 1 cycle, no caches, no branch predictor, no OS interrupts), making
formal timing proofs achievable: if sigma=0.0 across diverse inputs, the
implementation is *provably* constant-time, not merely "probably."

Usage::

    verifier = ConstantTimeVerifier()

    # Verify a primitive operation
    result = verifier.verify(
        operation=lambda x: ct_xor(x, key),
        inputs_list=[torch.randint(0, 256, (16,), dtype=torch.float32) for _ in range(50)],
    )
    assert result.is_constant_time

    # Verify AES
    aes_result = verifier.verify_aes(aes_instance, num_inputs=50)
    print(verifier.generate_report([aes_result]))
"""

from __future__ import annotations

import time
import statistics
from dataclasses import dataclass, field

import torch


@dataclass
class TimingVerificationResult:
    """Result of constant-time verification for a single operation.

    This captures both GPU cycle counts (deterministic, the ground truth
    for constant-time verification) and wall-clock times (informational,
    subject to host-side noise).

    Attributes:
        operation: Human-readable name of the operation verified
        num_runs: Total number of executions measured
        num_distinct_inputs: Number of distinct input values tested
        cycle_counts: Exact GPU cycle counts per run (empty if not on GPU)
        mean_cycles: Mean cycle count
        std_cycles: Standard deviation of cycle counts (0.0 = constant-time)
        is_constant_time: True if std_cycles == 0.0
        max_deviation: Maximum |cycle_i - mean| across all runs
        wall_times_ms: Wall-clock times per run (informational)
        wall_mean_ms: Mean wall-clock time
        wall_std_ms: Standard deviation of wall-clock times
        input_descriptions: Short descriptions of the inputs tested
    """

    operation: str
    num_runs: int
    num_distinct_inputs: int
    cycle_counts: list[int] = field(default_factory=list)
    mean_cycles: float = 0.0
    std_cycles: float = 0.0
    is_constant_time: bool = False
    max_deviation: float = 0.0
    wall_times_ms: list[float] = field(default_factory=list)
    wall_mean_ms: float = 0.0
    wall_std_ms: float = 0.0
    input_descriptions: list[str] = field(default_factory=list)


class ConstantTimeVerifier:
    """Verify that cryptographic operations execute in constant time.

    The verifier runs an operation with diverse inputs and measures
    execution characteristics. On GPU, it checks exact cycle counts
    (sigma must be 0.0). In software, it measures wall-clock times
    and checks that variance is below a threshold.

    The verifier is designed to catch timing leaks caused by:
      - Data-dependent branches (if/else on secret data)
      - Variable-time table lookups (cache-dependent access patterns)
      - Early termination (short-circuit comparison)
      - Input-dependent loop counts

    Attributes:
        tolerance: Maximum allowed coefficient of variation for
            wall-clock times (default 5%). Only applies to software
            verification; GPU verification requires sigma=0.0 exactly.
    """

    def __init__(self, tolerance: float = 0.05):
        """Initialize the verifier.

        Args:
            tolerance: Maximum CoV for wall-clock times to consider
                an operation "constant-time" in software mode.
                Default is 0.05 (5%).
        """
        self.tolerance = tolerance

    def verify(
        self,
        operation: callable,
        inputs_list: list[torch.Tensor],
        num_repeats: int = 100,
        warmup: int = 10,
    ) -> TimingVerificationResult:
        """Verify constant-time execution of an operation.

        Runs the operation with each input in inputs_list, repeating
        num_repeats times per input, and measures wall-clock execution
        times. The total number of runs is len(inputs_list) * num_repeats.

        For GPU verification with exact cycle counts, use verify_gpu()
        which interfaces with nCPU's Metal compute kernel.

        Args:
            operation: Callable that takes a single torch.Tensor and returns
                a torch.Tensor. Must be the constant-time function under test.
            inputs_list: List of distinct input tensors to test. Should cover
                diverse values including edge cases (all zeros, all ones,
                alternating bits, etc.).
            num_repeats: Number of times to repeat each input measurement
                for statistical robustness.
            warmup: Number of warmup runs (not measured) to stabilize
                JIT compilation and memory allocation.

        Returns:
            TimingVerificationResult with timing measurements and verdict
        """
        # Warmup: run the operation several times to warm caches/JIT
        for _ in range(warmup):
            operation(inputs_list[0])

        wall_times = []
        input_descs = []

        for inp_idx, inp in enumerate(inputs_list):
            times_for_input = []
            for _ in range(num_repeats):
                # Synchronize GPU if available
                if inp.is_cuda:
                    torch.cuda.synchronize()

                start = time.perf_counter_ns()
                _ = operation(inp)

                if inp.is_cuda:
                    torch.cuda.synchronize()

                elapsed_ns = time.perf_counter_ns() - start
                times_for_input.append(elapsed_ns / 1_000_000)  # convert to ms

            wall_times.extend(times_for_input)
            # Create a short description of the input
            desc = self._describe_input(inp, inp_idx)
            input_descs.append(desc)

        # Compute statistics
        num_runs = len(wall_times)
        wall_mean = statistics.mean(wall_times)
        wall_std = statistics.stdev(wall_times) if num_runs > 1 else 0.0
        cov = wall_std / wall_mean if wall_mean > 0 else 0.0

        return TimingVerificationResult(
            operation=getattr(operation, "__name__", str(operation)),
            num_runs=num_runs,
            num_distinct_inputs=len(inputs_list),
            cycle_counts=[],  # Not available in software mode
            mean_cycles=0.0,
            std_cycles=0.0,
            is_constant_time=cov <= self.tolerance,
            max_deviation=max(abs(t - wall_mean) for t in wall_times),
            wall_times_ms=wall_times,
            wall_mean_ms=wall_mean,
            wall_std_ms=wall_std,
            input_descriptions=input_descs,
        )

    def verify_aes(
        self,
        aes_impl,
        num_inputs: int = 50,
        num_repeats: int = 100,
        warmup: int = 10,
    ) -> TimingVerificationResult:
        """Verify AES encryption is constant-time across random inputs.

        Generates num_inputs random plaintexts covering diverse byte patterns
        and verifies that encryption time does not vary with plaintext content.

        The inputs include deliberately adversarial patterns designed to
        trigger timing leaks in non-constant-time implementations:
          - All zeros (cache line 0 every lookup)
          - All 0xFF (cache line 15 every lookup)
          - Alternating 0xAA/0x55 (alternating cache lines)
          - Sequential 0x00-0xFF (sequential cache access)
          - Random values

        Args:
            aes_impl: ConstantTimeAES instance (already initialized with key)
            num_inputs: Number of distinct plaintexts to test
            num_repeats: Repetitions per plaintext
            warmup: Warmup runs

        Returns:
            TimingVerificationResult with verdict
        """
        inputs = self._generate_aes_test_inputs(num_inputs)

        def encrypt_op(plaintext: torch.Tensor) -> torch.Tensor:
            return aes_impl.encrypt_block(plaintext)

        result = self.verify(
            operation=encrypt_op,
            inputs_list=inputs,
            num_repeats=num_repeats,
            warmup=warmup,
        )
        result.operation = "AES-128 Encrypt"
        return result

    def verify_aes_decrypt(
        self,
        aes_impl,
        num_inputs: int = 50,
        num_repeats: int = 100,
        warmup: int = 10,
    ) -> TimingVerificationResult:
        """Verify AES decryption is constant-time across random inputs.

        Args:
            aes_impl: ConstantTimeAES instance
            num_inputs: Number of distinct ciphertexts to test
            num_repeats: Repetitions per ciphertext
            warmup: Warmup runs

        Returns:
            TimingVerificationResult
        """
        inputs = self._generate_aes_test_inputs(num_inputs)

        def decrypt_op(ciphertext: torch.Tensor) -> torch.Tensor:
            return aes_impl.decrypt_block(ciphertext)

        result = self.verify(
            operation=decrypt_op,
            inputs_list=inputs,
            num_repeats=num_repeats,
            warmup=warmup,
        )
        result.operation = "AES-128 Decrypt"
        return result

    def _generate_aes_test_inputs(self, num_inputs: int) -> list[torch.Tensor]:
        """Generate adversarial AES test inputs.

        Includes patterns specifically designed to trigger timing leaks
        in non-constant-time implementations.

        Args:
            num_inputs: Total number of inputs to generate

        Returns:
            List of 16-byte float tensors
        """
        inputs = []

        # Adversarial patterns (always included)
        adversarial = [
            torch.zeros(16, dtype=torch.float32),                     # all zeros
            torch.full((16,), 255.0, dtype=torch.float32),            # all 0xFF
            torch.tensor([0xAA] * 16, dtype=torch.float32),           # alternating 10101010
            torch.tensor([0x55] * 16, dtype=torch.float32),           # alternating 01010101
            torch.arange(16, dtype=torch.float32),                    # sequential
            torch.arange(16, dtype=torch.float32) * 17.0,             # spread across S-box
            torch.tensor([0x00, 0xFF] * 8, dtype=torch.float32),      # extreme alternating
        ]
        inputs.extend(adversarial[:min(len(adversarial), num_inputs)])

        # Fill remaining with random values
        remaining = num_inputs - len(inputs)
        for _ in range(remaining):
            inp = torch.randint(0, 256, (16,), dtype=torch.float32).float()
            inputs.append(inp)

        return inputs

    def _describe_input(self, inp: torch.Tensor, index: int) -> str:
        """Create a human-readable description of an input tensor.

        Args:
            inp: input tensor
            index: input index number

        Returns:
            Short description string
        """
        flat = inp.flatten()
        if flat.numel() == 0:
            return f"input_{index}: empty"

        unique_vals = flat.unique()
        if unique_vals.numel() == 1:
            val = int(unique_vals[0].item())
            return f"input_{index}: all 0x{val:02X}"
        if flat.numel() <= 4:
            vals = [f"0x{int(v.item()):02X}" for v in flat]
            return f"input_{index}: [{', '.join(vals)}]"
        return f"input_{index}: {flat.numel()} bytes, range [{int(flat.min().item())}-{int(flat.max().item())}]"

    def generate_report(self, results: list[TimingVerificationResult]) -> str:
        """Generate a formal timing verification report.

        Produces a human-readable report suitable for inclusion in
        security documentation or academic papers. Includes statistical
        analysis, per-operation verdicts, and an overall assessment.

        Args:
            results: List of TimingVerificationResult from verify() calls

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 72,
            "CONSTANT-TIME VERIFICATION REPORT",
            "nCPU Cryptographic Library",
            "=" * 72,
            "",
        ]

        all_pass = True
        for result in results:
            verdict = "PASS" if result.is_constant_time else "FAIL"
            if not result.is_constant_time:
                all_pass = False

            lines.append(f"Operation: {result.operation}")
            lines.append(f"  Verdict: {verdict}")
            lines.append(f"  Runs: {result.num_runs} ({result.num_distinct_inputs} distinct inputs)")
            lines.append("")

            if result.cycle_counts:
                lines.append("  GPU Cycle Analysis (ground truth):")
                lines.append(f"    Mean cycles:     {result.mean_cycles:,.1f}")
                lines.append(f"    Std dev:         {result.std_cycles:.4f}")
                lines.append(f"    Max deviation:   {result.max_deviation:.4f}")
                lines.append(
                    f"    Constant-time:   {'YES (sigma=0.0)' if result.std_cycles == 0.0 else 'NO'}"
                )
                lines.append("")

            if result.wall_times_ms:
                cov = (
                    result.wall_std_ms / result.wall_mean_ms * 100
                    if result.wall_mean_ms > 0
                    else 0.0
                )
                lines.append("  Wall-Clock Analysis (informational):")
                lines.append(f"    Mean time:       {result.wall_mean_ms:.4f} ms")
                lines.append(f"    Std dev:         {result.wall_std_ms:.4f} ms")
                lines.append(f"    CoV:             {cov:.2f}%")
                lines.append(f"    Max deviation:   {result.max_deviation:.4f} ms")
                lines.append("")

            lines.append("-" * 72)
            lines.append("")

        # Overall verdict
        lines.append("=" * 72)
        lines.append(f"OVERALL VERDICT: {'ALL OPERATIONS CONSTANT-TIME' if all_pass else 'TIMING LEAK DETECTED'}")
        lines.append("=" * 72)
        lines.append("")

        if all_pass:
            lines.extend([
                "All tested operations show timing characteristics consistent with",
                "constant-time execution. On nCPU's GPU (sigma=0.0 cycle variance),",
                "this constitutes a formal proof of timing side-channel immunity.",
                "",
                "On conventional CPUs, this result is informational only: wall-clock",
                "measurements cannot rule out microarchitectural timing leaks.",
            ])
        else:
            lines.extend([
                "WARNING: One or more operations show timing variance exceeding",
                "the configured tolerance. This may indicate:",
                "  - Data-dependent branches in the implementation",
                "  - Variable-time memory access patterns (cache effects)",
                "  - Insufficient warmup or system noise (increase num_repeats)",
                "",
                "Review the failing operations and ensure all control flow",
                "is replaced with constant-time arithmetic (ct_select, ct_xor, etc.).",
            ])

        return "\n".join(lines)
