"""
Mog Language Integration for SOME

Mog (moglang.org) is a programming language designed for AI Agents with:
- Capability-based security
- Async primitives (spawn, await, all, race)
- Tensor support
- Embedding API for host applications

This module integrates Mog with SOME for agentic code generation.
"""

import subprocess
import tempfile
import os
from dataclasses import dataclass
from typing import Optional, Callable
from pathlib import Path


@dataclass
class MogConfig:
    """Configuration for Mog integration"""
    mog_path: str = "mog"  # Path to mog compiler
    timeout: int = 30  # Compilation/execution timeout
    sandboxed: bool = True  # Run in sandbox


class MogProvider:
    """
    LLM provider that generates Mog code instead of Python.

    Mog is designed for AI agents with:
    - Capability-based security
    - Async programming primitives
    - Tensor operations
    - Safe I/O through capabilities
    """

    def __init__(self, config: Optional[MogConfig] = None):
        self.config = config or MogConfig()
        self._check_mog_installed()

    def _check_mog_installed(self):
        """Check if Mog is installed"""
        try:
            result = subprocess.run(
                [self.config.mog_path, "--version"],
                capture_output=True,
                timeout=5,
            )
            self.version = result.stdout.decode() if result.returncode == 0 else "unknown"
        except:
            self.version = None

    def generate(self, prompt: str) -> str:
        """
        Generate Mog code from prompt.

        In production, this would call an LLM that outputs Mog code.
        For now, returns example Mog code.
        """
        # Example: translate Python prompt to Mog
        # In real implementation: use LLM to generate Mog

        examples = {
            "fibonacci": self._fib_mog(),
            "factorial": self._factorial_mog(),
            "hello": self._hello_mog(),
            "async": self._async_mog(),
        }

        for key, code in examples.items():
            if key in prompt.lower():
                return code

        # Default: return example
        return self._hello_mog()

    def _fib_mog(self) -> str:
        """Fibonacci in Mog"""
        return '''
pub fn fib(n: int) -> int {
    if n <= 1 {
        return n
    }
    return fib(n - 1) + fib(n - 2)
}

pub fn main() {
    let result = fib(10)
    print("fib(10) = ")
    print(result)
}
'''

    def _factorial_mog(self) -> str:
        """Factorial in Mog"""
        return '''
pub fn factorial(n: int) -> int {
    if n <= 1 {
        return 1
    }
    return n * factorial(n - 1)
}

pub fn main() {
    let result = factorial(5)
    print("5! = ")
    print(result)
}
'''

    def _hello_mog(self) -> str:
        """Hello world in Mog"""
        return '''
pub fn main() {
    print("Hello from Mog!")
}
'''

    def _async_mog(self) -> str:
        """Async example in Mog"""
        return '''
pub async fn fetch_data(url: string) -> string {
    // Async fetch - Mog supports spawn/await
    return "data"
}

pub async fn main() {
    let result = await fetch_data("https://example.com")
    print(result)
}
'''


class MogExecutor:
    """
    Executes Mog code and returns results.
    """

    def __init__(self, config: Optional[MogConfig] = None):
        self.config = config or MogConfig()

    def execute(self, code: str) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Execute Mog code.

        Returns: (success, stdout, stderr)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write code to temp file
            src_path = Path(tmpdir) / "main.mog"
            src_path.write_text(code)

            try:
                # Compile and run
                result = subprocess.run(
                    [self.config.mog_path, "run", str(src_path)],
                    capture_output=True,
                    timeout=self.config.timeout,
                    cwd=tmpdir,
                )

                success = result.returncode == 0
                stdout = result.stdout.decode() if result.stdout else ""
                stderr() if result.stderr = result.stderr.decode else ""

                return success, stdout, stderr

            except subprocess.TimeoutExpired:
                return False, "", "Timeout"
            except Exception as e:
                return False, "", str(e)


class MogVerifier:
    """
    Verifies Mog code execution against test cases.
    """

    def __init__(self):
        self.executor = MogExecutor()

    def verify(self, code: str, expected_output: str) -> tuple[bool, Optional[str]]:
        """Verify Mog code produces expected output"""
        success, stdout, stderr = self.executor.execute(code)

        if not success:
            return False, stderr

        if expected_output.strip() in stdout.strip():
            return True, None
        else:
            return False, f"Expected: {expected_output}, Got: {stdout}"


class MogSOMEAgent:
    """
    SOME agent that generates and verifies Mog code.

    Combines:
    - LLM for generating Mog code
    - MogExecutor for running code
    - Feedback system for self-improvement
    """

    def __init__(
        self,
        llm_provider: Optional[Callable] = None,
        mog_provider: Optional[MogProvider] = None,
    ):
        self.llm_provider = llm_provider
        self.mog_provider = mog_provider or MogProvider()
        self.verifier = MogVerifier()
        self.history = []

    def generate_and_verify(
        self,
        prompt: str,
        expected_output: Optional[str] = None,
        max_retries: int = 3,
    ) -> dict:
        """Generate Mog code and verify it works"""

        for attempt in range(max_retries):
            # Generate code
            if self.llm_provider:
                code = self.llm_provider(prompt)
            else:
                code = self.mog_provider.generate(prompt)

            # Verify
            if expected_output:
                success, error = self.verifier.verify(code, expected_output)
            else:
                success, stdout, error = self.mog_provider.generate(prompt)
                success = stdout is not None

            result = {
                "attempt": attempt + 1,
                "code": code,
                "success": success,
                "error": error,
            }

            self.history.append(result)

            if success:
                return result

            # Modify prompt for retry
            if attempt < max_retries - 1:
                prompt = f"{prompt}\nPrevious attempt failed: {error}\nFix the code."

        return self.history[-1]


def demo():
    """Demo of Mog integration"""
    print("=" * 60)
    print("Mog Language Integration Demo")
    print("=" * 60)

    # Check if Mog is installed
    provider = MogProvider()
    if provider.version:
        print(f"Mog version: {provider.version}")
    else:
        print("Mog not installed - using mock generation")

    # Generate Mog code
    print("\n--- Generating Mog code ---")

    for task in ["fibonacci", "factorial", "hello", "async"]:
        code = provider.generate(f"Write {task} in Mog")
        print(f"\n{task}:")
        print(code[:200] + "..." if len(code) > 200 else code)

    # Verify (will fail if Mog not installed)
    print("\n--- Verifying Mog code ---")
    verifier = MogVerifier()

    hello_code = provider.generate("hello")
    success, error = verifier.verify(hello_code, "Hello from Mog")

    print(f"Hello world verification: {success}")
    if error:
        print(f"Error: {error}")

    # Test agent
    print("\n--- Testing Mog SOME Agent ---")
    agent = MogSOMEAgent()
    result = agent.generate_and_verify(
        "Write a function that calculates fibonacci",
        expected_output="55"  # fib(10) = 55
    )

    print(f"Attempt: {result['attempt']}")
    print(f"Success: {result['success']}")
    if result['error']:
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    demo()
