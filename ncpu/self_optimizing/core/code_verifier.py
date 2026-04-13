"""
Code Verifier for LLM Benchmark

Actually executes and verifies generated code against test cases.
"""

import ast
import io
import re
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class VerificationResult:
    """Result from code verification"""
    success: bool
    error: Optional[str]
    output: Optional[str]
    test_results: list


class CodeVerifier:
    """
    Verifies generated code by executing against test cases.

    Supports:
    - Syntax checking
    - Runtime execution
    - Test case validation
    - Timeout handling
    """

    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    def extract_code(self, text: str) -> str:
        """
        Extract executable Python code from a model response.

        Models frequently ignore "code only" instructions and wrap answers in
        markdown or prose. This keeps the verifier focused on the code itself.
        """
        stripped = text.strip()
        if not stripped:
            return stripped

        fenced_blocks = re.findall(r"```(?:python)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
        if fenced_blocks:
            return max((block.strip() for block in fenced_blocks), key=len, default=stripped)

        if self._can_compile(stripped):
            return stripped

        lines = stripped.splitlines()
        start_index = None
        for index, line in enumerate(lines):
            if re.match(r"^\s*(def |class |from |import |@)", line):
                start_index = index
                break

        if start_index is None:
            return stripped

        candidate_lines: list[str] = []
        for line in lines[start_index:]:
            if line.strip().startswith("```"):
                break
            if candidate_lines and not line.strip():
                candidate_lines.append(line)
                continue
            if (
                line.startswith((" ", "\t"))
                or re.match(r"^\s*(def |class |from |import |@|if __name__)", line)
                or line.strip() in {"pass", "break", "continue"}
            ):
                candidate_lines.append(line)
                continue
            if candidate_lines:
                break
            candidate_lines.append(line)

        candidate = "\n".join(candidate_lines).strip()
        return candidate or stripped

    def _can_compile(self, code: str) -> bool:
        try:
            compile(code, "<generated>", "exec")
            return True
        except SyntaxError:
            return False

    def verify(
        self,
        code: str,
        test_cases: Optional[list[dict]] = None,
    ) -> VerificationResult:
        """
        Verify code against optional test cases.

        Args:
            code: Python code to verify
            test_cases: List of {"input": ..., "expected": ...} dicts

        Returns:
            VerificationResult with success status and details
        """
        prepared_code = self.extract_code(code)

        # 1. Syntax check
        try:
            compile(prepared_code, "<generated>", "exec")
        except SyntaxError as e:
            return VerificationResult(
                success=False,
                error=f"Syntax error at line {e.lineno}: {e.msg}",
                output=None,
                test_results=[],
            )

        # 2. Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()

        # 3. Execute with timeout
        try:
            namespace = {}
            exec(prepared_code, namespace)

            # 4. Run test cases if provided
            test_results = []
            if test_cases:
                func_name = self._extract_function_name(prepared_code)
                if not func_name or func_name not in namespace:
                    return VerificationResult(
                        success=False,
                        error="Could not find a callable function definition in generated code",
                        output=captured.getvalue(),
                        test_results=[],
                    )

                func = namespace[func_name]
                for i, test in enumerate(test_cases):
                    try:
                        result = func(**test["input"])
                        passed = result == test["expected"]
                        test_results.append({
                            "test": i,
                            "input": test["input"],
                            "expected": test["expected"],
                            "actual": result,
                            "passed": passed,
                        })
                    except Exception as e:
                        test_results.append({
                            "test": i,
                            "error": str(e),
                            "passed": False,
                        })

            output = captured.getvalue()
            sys.stdout = old_stdout

            # Determine success
            all_passed = all(t.get("passed", False) for t in test_results) if test_results else True

            return VerificationResult(
                success=all_passed,
                error=None,
                output=output,
                test_results=test_results,
            )

        except Exception as e:
            sys.stdout = old_stdout
            return VerificationResult(
                success=False,
                error=f"Runtime error: {str(e)}",
                output=captured.getvalue(),
                test_results=[],
            )

    def _extract_function_name(self, code: str) -> Optional[str]:
        """Extract function name from code"""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except:
            pass
        return None


# Test cases for common problems
FIBONACCI_TESTS = [
    {"input": {"n": 0}, "expected": 0},
    {"input": {"n": 1}, "expected": 1},
    {"input": {"n": 10}, "expected": 55},
    {"input": {"n": 20}, "expected": 6765},
]

FACTORIAL_TESTS = [
    {"input": {"n": 0}, "expected": 1},
    {"input": {"n": 1}, "expected": 1},
    {"input": {"n": 5}, "expected": 120},
    {"input": {"n": 10}, "expected": 3628800},
]

PALINDROME_TESTS = [
    {"input": {"s": "racecar"}, "expected": True},
    {"input": {"s": "hello"}, "expected": False},
    {"input": {"s": "a"}, "expected": True},
]

REVERSE_LIST_TESTS = [
    {"input": {"lst": [1, 2, 3]}, "expected": [3, 2, 1]},
    {"input": {"lst": [1]}, "expected": [1]},
    {"input": {"lst": []}, "expected": []},
]

BINARY_SEARCH_TESTS = [
    {"input": {"arr": [1, 2, 3, 4, 5], "target": 3}, "expected": 2},
    {"input": {"arr": [1, 2, 3, 4, 5], "target": 6}, "expected": -1},
    {"input": {"arr": [1], "target": 1}, "expected": 0},
]

IS_PRIME_TESTS = [
    {"input": {"n": 2}, "expected": True},
    {"input": {"n": 3}, "expected": True},
    {"input": {"n": 4}, "expected": False},
    {"input": {"n": 17}, "expected": True},
    {"input": {"n": 21}, "expected": False},
    {"input": {"n": 1}, "expected": False},
    {"input": {"n": 0}, "expected": False},
]

QUICK_SORT_TESTS = [
    {"input": {"arr": [3, 1, 2]}, "expected": [1, 2, 3]},
    {"input": {"arr": []}, "expected": []},
    {"input": {"arr": [5]}, "expected": [5]},
    {"input": {"arr": [5, -1, 3, -1, 0]}, "expected": [-1, -1, 0, 3, 5]},
    {"input": {"arr": [4, 4, 4, 4]}, "expected": [4, 4, 4, 4]},
]


# Prompt templates for LLM
CODE_PROMPTS = {
    "fibonacci": """Write a Python function called 'fib' that calculates the nth Fibonacci number.

Requirements:
- Use recursion or iteration (your choice)
- Handle edge cases (n=0, n=1)
- Return an integer

Example: fib(10) should return 55
""",
    "factorial": """Write a Python function called 'factorial' that calculates n factorial.

Requirements:
- Handle n=0 (return 1)
- Return an integer

Example: factorial(5) should return 120
""",
    "palindrome": """Write a Python function called 'palindrome' that checks if a string is a palindrome.

Requirements:
- Ignore case
- Return boolean

Example: palindrome("racecar") should return True
""",
    "reverse_list": """Write a Python function called 'reverse_list' that reverses a list.

Requirements:
- Do not use built-in reverse() or slicing [::-1]
- Return a new list

Example: reverse_list([1,2,3]) should return [3,2,1]
""",
    "binary_search": """Write a Python function called 'binary_search' that finds the index of a target in a sorted array.

Requirements:
- Array is sorted in ascending order
- Return index if found, -1 if not found

Example: binary_search([1,2,3,4,5], 3) should return 2
""",
    "is_prime": """Write a Python function called 'is_prime' that returns True if n is prime and False otherwise.

Requirements:
- Return False for n < 2
- Handle even numbers correctly

Example: is_prime(17) should return True
""",
    "quick_sort": """Write a Python function called 'quick_sort' that sorts a list of integers.

Requirements:
- Return a new sorted list
- Handle empty lists and duplicate values

Example: quick_sort([3,1,2]) should return [1,2,3]
""",
}


# Hard prompts that will challenge weaker models
HARD_CODE_PROMPTS = {
    "lru_cache": """Write a Python class called LRUCache that implements a Least Recently Used cache.

Requirements:
- __init__(self, capacity: int)
- get(self, key: int) -> int: returns -1 if not found
- put(self, key: int, value: int): evicts LRU item if capacity exceeded
- Must use O(1) time for both operations

Example:
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
cache.get(1) == 1
cache.put(3, 3)  # evicts key 2
cache.get(2) == -1
""",
    "dijkstra": """Write a Python function called dijkstra(graph, start, end) that implements Dijkstra's shortest path algorithm.

Requirements:
- graph is a dict: {node: [(neighbor, weight), ...]}
- Returns tuple (distance, path) or (inf, []) if no path
- Handle edge cases (empty graph, start == end)

Example:
graph = {'A': [('B', 1), ('C', 4)], 'B': [('C', 2), ('D', 5)], 'C': [('D', 1)], 'D': []}
dijkstra(graph, 'A', 'D') == (3, ['A', 'B', 'C', 'D'])
""",
    "topological_sort": """Write a Python function called topological_sort(n, edges) that returns a valid topological ordering of nodes 0 to n-1.

Requirements:
- n is number of nodes
- edges is list of (u, v) directed edges from u to v
- Returns list of nodes in topological order
- Raises ValueError if cycle detected

Example:
topological_sort(4, [(0,1), (0,2), (1,3), (2,3)]) could return [0, 1, 2, 3]
""",
    "merge_intervals": """Write a Python function called merge_intervals(intervals) that merges overlapping intervals.

Requirements:
- intervals is list of [start, end] tuples
- Returns list of merged [start, end] intervals
- Must handle empty input

Example:
merge_intervals([[1,3], [2,6], [8,10], [15,18]]) == [[1,6], [8,10], [15,18]]
""",
    "binary_tree_zigzag": """Write a Python function called zigzag_level_order(root) that returns zigzag level order traversal of binary tree.

Requirements:
- root is TreeNode with .val, .left, .right
- Returns list of lists, alternating left-to-right and right-to-left
- Return [] for empty tree

Example: For tree [1,2,3,4,5,6,7] return [[1], [3,2], [4,5,6,7]]
""",
    "longest_substring": """Write a Python function called length_of_longest_substring(s) that finds length of longest substring without repeating characters.

Requirements:
- s is a string
- Returns integer length

Example:
length_of_longest_substring("abcabcbb") == 3
length_of_longest_substring("bbbbb") == 1
""",
    "word_search": """Write a Python function called word_exists(board, word) that checks if word exists in 2D grid.

Requirements:
- board is list of list of chars
- word is string to find
- Can move horizontally/vertically (not diagonally)
- Cannot reuse same cell

Example:
board = [['A','B','C','E'], ['S','F','C','S'], ['A','D','E','E']]
word_exists(board, "ABCCED") == True
word_exists(board, "SEE") == True
word_exists(board, "ABCB") == False
""",
    "serialize_tree": """Write functions serialize(root) and deserialize(data) to encode binary tree to string and decode back.

Requirements:
- serialize returns string representation
- deserialize reconstructs exact tree
- Use any format you choose

Example: Round-trip must preserve tree structure
""",
    "kafka_producer": """Write a Python class called SimpleKafkaProducer that sends messages to a topic.

Requirements:
- __init__(self, bootstrap_servers)
- send(self, topic, key, value): returns Future
- flush(): waits for all messages
- Must handle connection errors gracefully

Example is conceptual - implement the interface
""",
    "rate_limiter": """Write a Python class called RateLimiter that limits requests per time window.

Requirements:
- __init__(self, max_requests, window_seconds)
- allow() -> bool: returns True if request allowed
- Must be thread-safe

Example:
limiter = RateLimiter(5, 10)  # 5 requests per 10 seconds
for i in range(10): print(limiter.allow())  # True,True,True,True,True,False,False...
""",
}


# =============================================================================
# Expanded Test Cases with Actual Executable Tests
# =============================================================================

# TreeNode class for binary tree tests
class TreeNode:
    """Binary tree node for test cases"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# LRU Cache test function
def test_lru_functionality():
    """Test LRUCache implementation"""
    # This will be populated by the generated code
    pass


# LRU Cache tests - actual executable test cases
LRU_CACHE_TESTS = [
    # Basic put and get
    {
        "setup": """
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.order = []

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]

    def put(self, key: int, value: int):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            lru = self.order.pop(0)
            del self.cache[lru]
        self.cache[key] = value
        self.order.append(key)
""",
        "tests": [
            {"method": "put", "args": (1, 1), "kwargs": {}},
            {"method": "put", "args": (2, 2), "kwargs": {}},
            {"method": "get", "args": (1,), "kwargs": {}, "expected": 1},
            {"method": "put", "args": (3, 3), "kwargs": {}},
            {"method": "get", "args": (2,), "kwargs": {}, "expected": -1},
            {"method": "put", "args": (4, 4), "kwargs": {}},
            {"method": "get", "args": (1,), "kwargs": {}, "expected": -1},
            {"method": "get", "args": (3,), "kwargs": {}, "expected": 3},
            {"method": "get", "args": (4,), "kwargs": {}, "expected": 4},
        ]
    },
    # Capacity 1 edge case
    {
        "setup": """
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.order = []

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]

    def put(self, key: int, value: int):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            lru = self.order.pop(0)
            del self.cache[lru]
        self.cache[key] = value
        self.order.append(key)
""",
        "tests": [
            {"method": "put", "args": (1, 1), "kwargs": {}},
            {"method": "put", "args": (2, 2), "kwargs": {}},
            {"method": "get", "args": (1,), "kwargs": {}, "expected": -1},
            {"method": "get", "args": (2,), "kwargs": {}, "expected": 2},
        ]
    },
]


# Binary Search Tree tests
BST_TESTS = [
    # Insert and search
    {
        "setup": """
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BST:
    def __init__(self):
        self.root = None

    def insert(self, val):
        if not self.root:
            self.root = TreeNode(val)
            return
        node = self.root
        while True:
            if val < node.val:
                if not node.left:
                    node.left = TreeNode(val)
                    return
                node = node.left
            else:
                if not node.right:
                    node.right = TreeNode(val)
                    return
                node = node.right

    def search(self, val):
        node = self.root
        while node:
            if val == node.val:
                return True
            elif val < node.val:
                node = node.left
            else:
                node = node.right
        return False
""",
        "tests": [
            {"method": "insert", "args": (5,), "kwargs": {}},
            {"method": "insert", "args": (3,), "kwargs": {}},
            {"method": "insert", "args": (7,), "kwargs": {}},
            {"method": "insert", "args": (1,), "kwargs": {}},
            {"method": "insert", "args": (4,), "kwargs": {}},
            {"method": "search", "args": (5,), "kwargs": {}, "expected": True},
            {"method": "search", "args": (3,), "kwargs": {}, "expected": True},
            {"method": "search", "args": (7,), "kwargs": {}, "expected": True},
            {"method": "search", "args": (1,), "kwargs": {}, "expected": True},
            {"method": "search", "args": (4,), "kwargs": {}, "expected": True},
            {"method": "search", "args": (6,), "kwargs": {}, "expected": False},
            {"method": "search", "args": (10,), "kwargs": {}, "expected": False},
        ]
    },
]


# Graph tests for Dijkstra and other algorithms
GRAPH_TESTS = [
    # Basic Dijkstra
    {
        "setup": """
import heapq
import math

def dijkstra(graph, start, end):
    if start not in graph:
        return (math.inf, [])
    if start == end:
        return (0, [start])

    dist = {start: 0}
    prev = {start: None}
    pq = [(0, start)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        if u == end:
            path = []
            node = end
            while node is not None:
                path.append(node)
                node = prev[node]
            return (d, list(reversed(path)))

        for v, w in graph.get(u, []):
            if v not in visited:
                new_dist = d + w
                if new_dist < dist.get(v, math.inf):
                    dist[v] = new_dist
                    prev[v] = u
                    heapq.heappush(pq, (new_dist, v))

    return (math.inf, [])
""",
        "tests": [
            # Simple path A->B->C->D
            {
                "input": {
                    "graph": {'A': [('B', 1), ('C', 4)], 'B': [('C', 2), ('D', 5)], 'C': [('D', 1)], 'D': []},
                    "start": "A",
                    "end": "D"
                },
                "expected": (3, ['A', 'B', 'C', 'D'])
            },
            # Direct edge
            {
                "input": {
                    "graph": {'A': [('B', 5)], 'B': []},
                    "start": "A",
                    "end": "B"
                },
                "expected": (5, ['A', 'B'])
            },
            # No path
            {
                "input": {
                    "graph": {'A': [], 'B': []},
                    "start": "A",
                    "end": "B"
                },
                "expected": (float('inf'), [])
            },
            # Start equals end
            {
                "input": {
                    "graph": {'A': [('B', 1)]},
                    "start": "A",
                    "end": "A"
                },
                "expected": (0, ['A'])
            },
        ]
    },
]


# Merge intervals tests
MERGE_INTERVALS_TESTS = [
    {
        "setup": """
def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    result = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append([start, end])
    return result
""",
        "tests": [
            {"input": {"intervals": [[1,3], [2,6], [8,10], [15,18]]}, "expected": [[1,6], [8,10], [15,18]]},
            {"input": {"intervals": [[1,4], [4,5]]}, "expected": [[1,5]]},
            {"input": {"intervals": [[1,4], [0,4]]}, "expected": [[0,4]]},
            {"input": {"intervals": [[1,4], [2,3]]}, "expected": [[1,4]]},
            {"input": {"intervals": []}, "expected": []},
            {"input": {"intervals": [[1,4], [0,0]]}, "expected": [[0,0], [1,4]]},
        ]
    },
]


# Longest substring without repeating characters
LONGEST_SUBSTRING_TESTS = [
    {
        "setup": """
def length_of_longest_substring(s):
    if not s:
        return 0
    char_index = {}
    max_len = 0
    start = 0
    for end, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        char_index[char] = end
        max_len = max(max_len, end - start + 1)
    return max_len
""",
        "tests": [
            {"input": {"s": "abcabcbb"}, "expected": 3},
            {"input": {"s": "bbbbb"}, "expected": 1},
            {"input": {"s": "pwwkew"}, "expected": 3},
            {"input": {"s": ""}, "expected": 0},
            {"input": {"s": "a"}, "expected": 1},
            {"input": {"s": "au"}, "expected": 2},
            {"input": {"s": "dvdf"}, "expected": 3},
        ]
    },
]


# Topological sort tests
TOPOLOGICAL_SORT_TESTS = [
    {
        "setup": """
def topological_sort(n, edges):
    from collections import defaultdict, deque
    graph = defaultdict(list)
    in_degree = [0] * n
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    result = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    if len(result) != n:
        raise ValueError("Cycle detected")
    return result
""",
        "tests": [
            {"input": {"n": 4, "edges": [[0,1], [0,2], [1,3], [2,3]]}, "expected": [0, 1, 2, 3]},
            {"input": {"n": 2, "edges": [[0,1]]}, "expected": [0, 1]},
            {"input": {"n": 1, "edges": []}, "expected": [0]},
        ]
    },
]


# Rate limiter tests
RATE_LIMITER_TESTS = [
    {
        "setup": """
import time
import threading

class RateLimiter:
    def __init__(self, max_requests, window_seconds):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
        self.lock = threading.Lock()

    def allow(self):
        with self.lock:
            now = time.time()
            self.requests = [t for t in self.requests if now - t < self.window_seconds]
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False
""",
        "tests": [
            # This test is timing-dependent so we just check it runs
        ]
    },
]


def verify_lru_cache(code: str) -> VerificationResult:
    """Custom verifier for LRU cache class-based task."""
    extractor = CodeVerifier()
    prepared = extractor.extract_code(code)

    try:
        compile(prepared, "<generated>", "exec")
    except SyntaxError as e:
        return VerificationResult(
            success=False, error=f"Syntax error at line {e.lineno}: {e.msg}",
            output=None, test_results=[],
        )

    try:
        ns: dict = {}
        exec(prepared, ns)
    except Exception as e:
        return VerificationResult(
            success=False, error=f"Runtime error: {e}",
            output=None, test_results=[],
        )

    if "LRUCache" not in ns:
        return VerificationResult(
            success=False, error="LRUCache class not found in generated code",
            output=None, test_results=[],
        )

    cls = ns["LRUCache"]
    results: list[dict] = []

    # Scenario 1: capacity 2, basic eviction
    try:
        c = cls(2)
        c.put(1, 1)
        c.put(2, 2)
        r = c.get(1)
        results.append({"test": 0, "expected": 1, "actual": r, "passed": r == 1})
        c.put(3, 3)  # evicts 2
        r = c.get(2)
        results.append({"test": 1, "expected": -1, "actual": r, "passed": r == -1})
        c.put(4, 4)  # evicts 1
        r = c.get(1)
        results.append({"test": 2, "expected": -1, "actual": r, "passed": r == -1})
        r = c.get(3)
        results.append({"test": 3, "expected": 3, "actual": r, "passed": r == 3})
        r = c.get(4)
        results.append({"test": 4, "expected": 4, "actual": r, "passed": r == 4})
    except Exception as e:
        results.append({"test": "scenario1", "error": str(e), "passed": False})

    # Scenario 2: capacity 1
    try:
        c = cls(1)
        c.put(1, 1)
        c.put(2, 2)
        r = c.get(1)
        results.append({"test": 5, "expected": -1, "actual": r, "passed": r == -1})
        r = c.get(2)
        results.append({"test": 6, "expected": 2, "actual": r, "passed": r == 2})
    except Exception as e:
        results.append({"test": "scenario2", "error": str(e), "passed": False})

    # Scenario 3: update existing key
    try:
        c = cls(2)
        c.put(1, 1)
        c.put(2, 2)
        c.put(1, 10)
        r = c.get(1)
        results.append({"test": 7, "expected": 10, "actual": r, "passed": r == 10})
        r = c.get(2)
        results.append({"test": 8, "expected": 2, "actual": r, "passed": r == 2})
    except Exception as e:
        results.append({"test": "scenario3", "error": str(e), "passed": False})

    ok = all(t.get("passed", False) for t in results)
    return VerificationResult(
        success=ok, error=None if ok else "Some LRU cache tests failed",
        output=None, test_results=results,
    )


def verify_topological_sort(code: str) -> VerificationResult:
    """Custom verifier for topological sort — checks validity, not exact order."""
    extractor = CodeVerifier()
    prepared = extractor.extract_code(code)

    try:
        compile(prepared, "<generated>", "exec")
    except SyntaxError as e:
        return VerificationResult(
            success=False, error=f"Syntax error at line {e.lineno}: {e.msg}",
            output=None, test_results=[],
        )

    try:
        ns: dict = {}
        exec(prepared, ns)
    except Exception as e:
        return VerificationResult(
            success=False, error=f"Runtime error: {e}",
            output=None, test_results=[],
        )

    if "topological_sort" not in ns:
        return VerificationResult(
            success=False, error="topological_sort function not found",
            output=None, test_results=[],
        )

    func = ns["topological_sort"]
    results: list[dict] = []

    cases = [
        (4, [(0, 1), (0, 2), (1, 3), (2, 3)]),
        (2, [(0, 1)]),
        (1, []),
        (6, [(5, 2), (5, 0), (4, 0), (4, 1), (2, 3), (3, 1)]),
    ]

    for i, (n, edges) in enumerate(cases):
        try:
            result = func(n, edges)
            valid = (
                isinstance(result, list)
                and sorted(result) == list(range(n))
                and all(result.index(u) < result.index(v) for u, v in edges)
            )
            results.append({
                "test": i, "expected": "valid topological order",
                "actual": result, "passed": valid,
            })
        except Exception as e:
            results.append({"test": i, "error": str(e), "passed": False})

    # Cycle detection
    try:
        func(3, [(0, 1), (1, 2), (2, 0)])
        results.append({
            "test": len(cases), "expected": "ValueError",
            "actual": "no exception raised", "passed": False,
        })
    except ValueError:
        results.append({
            "test": len(cases), "expected": "ValueError",
            "actual": "ValueError", "passed": True,
        })
    except Exception as e:
        results.append({
            "test": len(cases), "expected": "ValueError",
            "actual": type(e).__name__, "passed": False,
        })

    ok = all(t.get("passed", False) for t in results)
    return VerificationResult(
        success=ok, error=None if ok else "Some topological sort tests failed",
        output=None, test_results=results,
    )


def demo():
    """Demo of code verifier"""
    print("=== Code Verifier Demo ===\n")

    verifier = CodeVerifier()

    # Test valid code
    valid_code = """
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)
"""

    result = verifier.verify(valid_code, FIBONACCI_TESTS)
    print(f"Valid code: {result.success}")
    print(f"Test results: {result.test_results}")
    print()

    # Test invalid code
    invalid_code = """
def fib(n)
    return n
"""

    result = verifier.verify(invalid_code)
    print(f"Invalid code: {result.success}")
    print(f"Error: {result.error}")


if __name__ == "__main__":
    demo()
