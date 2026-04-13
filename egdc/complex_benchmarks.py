"""complex_benchmarks.py — Algorithm-complexity benchmarks for LLM synthesis.

These problems are deliberately beyond gradient-only synthesis: they require
sorting, dynamic programming, graph traversal, string manipulation, and number
theory — none of which can be expressed as a single differentiable expression.

Each benchmark has at least 6 diverse test cases including edge cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ComplexBenchmark:
    name: str
    category: str
    description: str
    fn_name: str
    signature: str  # Python function signature as a string
    examples: list[tuple[tuple[Any, ...], Any]]  # (args_tuple, expected_output)
    reference_solution: str  # correct Python code


# ---------------------------------------------------------------------------
# Reference solutions (verified correct)
# ---------------------------------------------------------------------------

_REF_BUBBLE_SORT = """\
def bubble_sort(arr):
    arr = list(arr)
    n = len(arr)
    for i in range(n):
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""

_REF_SELECTION_SORT = """\
def selection_sort(arr):
    arr = list(arr)
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
"""

_REF_INSERTION_SORT = """\
def insertion_sort(arr):
    arr = list(arr)
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
"""

_REF_COUNT_SORT = """\
def count_sort_small(arr):
    if not arr:
        return []
    counts = [0] * 10
    for x in arr:
        counts[x] += 1
    result = []
    for i in range(10):
        result.extend([i] * counts[i])
    return result
"""

_REF_SORTED_MERGE = """\
def sorted_merge(a, b):
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result
"""

_REF_BINARY_SEARCH = """\
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
"""

_REF_ROTATE_ARRAY = """\
def rotate_array(arr, k):
    if not arr:
        return []
    n = len(arr)
    k = k % n
    return arr[k:] + arr[:k]
"""

_REF_LONGEST_INCREASING = """\
def longest_increasing(arr):
    if not arr:
        return 0
    n = len(arr)
    dp = [1] * n
    for i in range(1, n):
        for j in range(i):
            if arr[j] < arr[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
"""

_REF_MAX_SUBARRAY = """\
def max_subarray(arr):
    if not arr:
        return 0
    max_sum = cur = arr[0]
    for x in arr[1:]:
        cur = max(x, cur + x)
        max_sum = max(max_sum, cur)
    return max_sum
"""

_REF_FIRST_DUPLICATE = """\
def first_duplicate(arr):
    seen = set()
    for x in arr:
        if x in seen:
            return x
        seen.add(x)
    return -1
"""

_REF_COUNT_INVERSIONS = """\
def count_inversions(arr):
    count = 0
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                count += 1
    return count
"""

_REF_TWO_SUM = """\
def two_sum_indices(arr, target):
    seen = {}
    for i, x in enumerate(arr):
        comp = target - x
        if comp in seen:
            j = seen[comp]
            return [j, i]
        seen[x] = i
    return [-1, -1]
"""

_REF_FLATTEN_CHUNKS = """\
def flatten_chunks(flat, chunk_sizes):
    result = []
    idx = 0
    for sz in chunk_sizes:
        result.extend(flat[idx:idx + sz])
        idx += sz
    return result
"""

_REF_SLIDING_WINDOW_MAX = """\
def sliding_window_max(arr, k):
    if not arr or k <= 0:
        return []
    result = []
    from collections import deque
    dq = deque()
    for i, x in enumerate(arr):
        while dq and arr[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if dq[0] < i - k + 1:
            dq.popleft()
        if i >= k - 1:
            result.append(arr[dq[0]])
    return result
"""

_REF_RUN_LENGTH_ENCODE = """\
def run_length_encode(s):
    if not s:
        return ""
    result = []
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            result.append(s[i - 1] + str(count))
            count = 1
    result.append(s[-1] + str(count))
    return "".join(result)
"""

_REF_RUN_LENGTH_DECODE = """\
def run_length_decode(s):
    result = []
    i = 0
    while i < len(s):
        char = s[i]
        i += 1
        num_str = ""
        while i < len(s) and s[i].isdigit():
            num_str += s[i]
            i += 1
        count = int(num_str) if num_str else 1
        result.append(char * count)
    return "".join(result)
"""

_REF_COUNT_WORDS = """\
def count_words(s):
    return len(s.split())
"""

_REF_REVERSE_WORDS = """\
def reverse_words(s):
    return " ".join(s.split()[::-1])
"""

_REF_CAESAR_CIPHER = """\
def caesar_cipher(s, k):
    result = []
    for c in s:
        if c.isalpha():
            base = ord('A') if c.isupper() else ord('a')
            result.append(chr((ord(c) - base + k) % 26 + base))
        else:
            result.append(c)
    return "".join(result)
"""

_REF_IS_PALINDROME_STR = """\
def is_palindrome_str(s):
    cleaned = "".join(c.lower() for c in s if c.isalpha())
    return 1 if cleaned == cleaned[::-1] else 0
"""

_REF_LONGEST_COMMON_PREFIX = """\
def longest_common_prefix(s):
    words = s.split()
    if not words:
        return ""
    prefix = words[0]
    for w in words[1:]:
        while not w.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix
"""

_REF_COMPRESS_STRING = """\
def compress_string(s):
    if not s:
        return ""
    result = [s[0]]
    for c in s[1:]:
        if c != result[-1]:
            result.append(c)
    return "".join(result)
"""

_REF_SIEVE_COUNT = """\
def sieve_count(n):
    if n < 2:
        return 0
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return sum(sieve)
"""

_REF_COUNT_DIVISORS = """\
def count_divisors(n):
    if n <= 0:
        return 0
    count = 0
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            count += 2
            if i * i == n:
                count -= 1
    return count
"""

_REF_PRIME_FACTORS_SUM = """\
def prime_factors_sum(n):
    if n <= 1:
        return 0
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return sum(factors)
"""

_REF_MODULAR_POW = """\
def modular_pow(a, b, m):
    if b == 0:
        return 1 % m
    result = 1
    a = a % m
    while b > 0:
        if b % 2 == 1:
            result = (result * a) % m
        a = (a * a) % m
        b //= 2
    return result
"""

_REF_DIGITAL_ROOT = """\
def digital_root(n):
    if n == 0:
        return 0
    return 1 + (n - 1) % 9
"""

_REF_CATALAN = """\
def catalan(n):
    if n <= 1:
        return 1
    result = 1
    for i in range(n):
        result = result * 2 * (2 * i + 1) // (i + 2)
    return result
"""

_REF_COIN_CHANGE_COUNT = """\
def coin_change_count(amount):
    coins = [1, 2, 5]
    dp = [0] * (amount + 1)
    dp[0] = 1
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] += dp[i - coin]
    return dp[amount]
"""

_REF_CLIMB_STAIRS = """\
def climb_stairs(n):
    if n <= 1:
        return 1
    a, b = 1, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
"""

_REF_MIN_COIN_CHANGE = """\
def min_coin_change(amount):
    coins = [1, 2, 5]
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
    return dp[amount] if dp[amount] != float('inf') else -1
"""

_REF_HOUSE_ROBBER = """\
def house_robber(arr):
    if not arr:
        return 0
    if len(arr) == 1:
        return arr[0]
    prev2, prev1 = 0, 0
    for x in arr:
        cur = max(prev1, prev2 + x)
        prev2, prev1 = prev1, cur
    return prev1
"""

_REF_TRIANGLE_PATH = """\
def triangle_path_sum(flat):
    # flat encodes triangle row by row: row 0 has 1 elem, row 1 has 2, etc.
    # Reconstruct rows
    rows = []
    idx = 0
    row_num = 0
    while idx < len(flat):
        row_size = row_num + 1
        rows.append(flat[idx:idx + row_size])
        idx += row_size
        row_num += 1
    if not rows:
        return 0
    n = len(rows)
    dp = list(rows[-1])
    for i in range(n - 2, -1, -1):
        for j in range(len(rows[i])):
            dp[j] = rows[i][j] + min(dp[j], dp[j + 1])
    return dp[0]
"""

_REF_BFS_DISTANCE = """\
def bfs_distance(encoded):
    # encoded: [n_nodes, n_edges, u1, v1, u2, v2, ..., src, dst]
    idx = 0
    n_nodes = encoded[idx]; idx += 1
    n_edges = encoded[idx]; idx += 1
    adj = [[] for _ in range(n_nodes)]
    for _ in range(n_edges):
        u = encoded[idx]; idx += 1
        v = encoded[idx]; idx += 1
        adj[u].append(v)
        adj[v].append(u)
    src = encoded[idx]; idx += 1
    dst = encoded[idx]
    if src == dst:
        return 0
    from collections import deque
    dist = [-1] * n_nodes
    dist[src] = 0
    q = deque([src])
    while q:
        node = q.popleft()
        for nb in adj[node]:
            if dist[nb] == -1:
                dist[nb] = dist[node] + 1
                if nb == dst:
                    return dist[nb]
                q.append(nb)
    return -1
"""

_REF_CONNECTED_COUNT = """\
def connected_count(encoded):
    # encoded: [n_nodes, n_edges, u1, v1, ...]
    idx = 0
    n_nodes = encoded[idx]; idx += 1
    n_edges = encoded[idx]; idx += 1
    parent = list(range(n_nodes))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for _ in range(n_edges):
        u = encoded[idx]; idx += 1
        v = encoded[idx]; idx += 1
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pu] = pv
    return len(set(find(i) for i in range(n_nodes)))
"""

_REF_GRAPH_DEGREE_SUM = """\
def graph_degree_sum(encoded):
    # encoded: [n_nodes, n_edges, u1, v1, ...]
    # returns sum of all node degrees (= 2 * n_edges for undirected)
    idx = 0
    n_nodes = encoded[idx]; idx += 1
    n_edges = encoded[idx]; idx += 1
    degree = [0] * n_nodes
    for _ in range(n_edges):
        u = encoded[idx]; idx += 1
        v = encoded[idx]; idx += 1
        degree[u] += 1
        degree[v] += 1
    return sum(degree)
"""

_REF_GRAPH_MAX_DEGREE = """\
def graph_max_degree(encoded):
    # encoded: [n_nodes, n_edges, u1, v1, ...]
    # returns maximum degree of any node
    idx = 0
    n_nodes = encoded[idx]; idx += 1
    n_edges = encoded[idx]; idx += 1
    degree = [0] * n_nodes
    for _ in range(n_edges):
        u = encoded[idx]; idx += 1
        v = encoded[idx]; idx += 1
        degree[u] += 1
        degree[v] += 1
    return max(degree) if n_nodes > 0 else 0
"""

_REF_GRAPH_HAS_CYCLE = """\
def graph_has_cycle(encoded):
    # encoded: [n_nodes, n_edges, u1, v1, ...]
    # returns 1 if undirected graph has a cycle, 0 otherwise
    idx = 0
    n_nodes = encoded[idx]; idx += 1
    n_edges = encoded[idx]; idx += 1
    parent = list(range(n_nodes))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for _ in range(n_edges):
        u = encoded[idx]; idx += 1
        v = encoded[idx]; idx += 1
        pu, pv = find(u), find(v)
        if pu == pv:
            return 1
        parent[pu] = pv
    return 0
"""

_REF_GRAPH_NEIGHBOR_COUNT = """\
def graph_neighbor_count(encoded):
    # encoded: [n_nodes, n_edges, u1, v1, ..., query_node]
    # returns number of neighbors of the query node
    idx = 0
    n_nodes = encoded[idx]; idx += 1
    n_edges = encoded[idx]; idx += 1
    adj = [0] * n_nodes
    for _ in range(n_edges):
        u = encoded[idx]; idx += 1
        v = encoded[idx]; idx += 1
        adj[u] += 1
        adj[v] += 1
    query = encoded[idx]
    return adj[query]
"""

# Additional array_algo benchmarks
_REF_PRODUCT_EXCEPT_SELF = """\
def product_except_self(arr):
    n = len(arr)
    if n == 0:
        return []
    result = [1] * n
    prefix = 1
    for i in range(n):
        result[i] = prefix
        prefix *= arr[i]
    suffix = 1
    for i in range(n - 1, -1, -1):
        result[i] *= suffix
        suffix *= arr[i]
    return result
"""

_REF_FIND_MISSING = """\
def find_missing(arr):
    # arr contains n-1 distinct integers from 1..n; find the missing one
    n = len(arr) + 1
    return n * (n + 1) // 2 - sum(arr)
"""

_REF_MAJORITY_ELEMENT = """\
def majority_element(arr):
    # Find element appearing more than n/2 times (Boyer-Moore voting)
    candidate = arr[0]
    count = 1
    for x in arr[1:]:
        if count == 0:
            candidate = x
            count = 1
        elif x == candidate:
            count += 1
        else:
            count -= 1
    return candidate
"""

# Additional number_theory benchmarks
_REF_IS_PERFECT = """\
def is_perfect(n):
    # 1 if n is a perfect number (sum of proper divisors == n), else 0
    if n <= 1:
        return 0
    s = 1
    i = 2
    while i * i <= n:
        if n % i == 0:
            s += i
            if i != n // i:
                s += n // i
        i += 1
    return 1 if s == n else 0
"""

_REF_GCD = """\
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a
"""


# ---------------------------------------------------------------------------
# Example generation helpers (run at module load)
# ---------------------------------------------------------------------------

def _exec_ref(code: str, fn_name: str, *args: Any) -> Any:
    """Execute a reference function and return its output."""
    ns: dict[str, Any] = {}
    exec(compile(code, "<ref>", "exec"), ns)
    return ns[fn_name](*args)


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------

COMPLEX_BENCHMARK_REGISTRY: list[ComplexBenchmark] = []


def _reg(b: ComplexBenchmark) -> ComplexBenchmark:
    COMPLEX_BENCHMARK_REGISTRY.append(b)
    return b


# ===========================================================================
# SORTING (5 benchmarks)
# ===========================================================================

_reg(ComplexBenchmark(
    name="bubble_sort",
    category="sorting",
    description="Sort an array of integers in ascending order using bubble sort.",
    fn_name="bubble_sort",
    signature="def bubble_sort(arr: list) -> list",
    examples=[
        (([3, 1, 2],), [1, 2, 3]),
        (([5, 4, 3, 2, 1],), [1, 2, 3, 4, 5]),
        (([1],), [1]),
        (([],), []),
        (([2, 2, 1],), [1, 2, 2]),
        (([10, -1, 5, 0, -3],), [-3, -1, 0, 5, 10]),
        (([7, 3, 9, 1, 6, 2],), [1, 2, 3, 6, 7, 9]),
    ],
    reference_solution=_REF_BUBBLE_SORT,
))

_reg(ComplexBenchmark(
    name="selection_sort",
    category="sorting",
    description="Sort an array of integers in ascending order using selection sort.",
    fn_name="selection_sort",
    signature="def selection_sort(arr: list) -> list",
    examples=[
        (([4, 2, 7, 1],), [1, 2, 4, 7]),
        (([],), []),
        (([5],), [5]),
        (([3, 3, 1, 2],), [1, 2, 3, 3]),
        (([9, 0, -5, 3],), [-5, 0, 3, 9]),
        (([1, 2, 3, 4, 5],), [1, 2, 3, 4, 5]),
        (([5, 4, 3, 2, 1, 0],), [0, 1, 2, 3, 4, 5]),
    ],
    reference_solution=_REF_SELECTION_SORT,
))

_reg(ComplexBenchmark(
    name="insertion_sort",
    category="sorting",
    description="Sort an array of integers in ascending order using insertion sort.",
    fn_name="insertion_sort",
    signature="def insertion_sort(arr: list) -> list",
    examples=[
        (([6, 2, 8, 1, 4],), [1, 2, 4, 6, 8]),
        (([],), []),
        (([42],), [42]),
        (([2, 1],), [1, 2]),
        (([3, 3, 3],), [3, 3, 3]),
        (([10, 5, 0, -5, -10],), [-10, -5, 0, 5, 10]),
        (([1, 3, 2, 4],), [1, 2, 3, 4]),
    ],
    reference_solution=_REF_INSERTION_SORT,
))

_reg(ComplexBenchmark(
    name="count_sort_small",
    category="sorting",
    description="Sort an array of integers (all values 0-9) using counting sort.",
    fn_name="count_sort_small",
    signature="def count_sort_small(arr: list) -> list",
    examples=[
        (([3, 1, 4, 1, 5, 9, 2, 6],), [1, 1, 2, 3, 4, 5, 6, 9]),
        (([],), []),
        (([0, 9, 5, 5, 0],), [0, 0, 5, 5, 9]),
        (([7],), [7]),
        (([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (([9, 8, 7, 6, 5, 4, 3, 2, 1, 0],), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (([3, 3, 3, 3],), [3, 3, 3, 3]),
    ],
    reference_solution=_REF_COUNT_SORT,
))

_reg(ComplexBenchmark(
    name="sorted_merge",
    category="sorting",
    description="Merge two already-sorted arrays into a single sorted array.",
    fn_name="sorted_merge",
    signature="def sorted_merge(a: list, b: list) -> list",
    examples=[
        (([1, 3, 5], [2, 4, 6]), [1, 2, 3, 4, 5, 6]),
        (([1, 2, 3], []), [1, 2, 3]),
        (([], [4, 5, 6]), [4, 5, 6]),
        (([], []), []),
        (([1], [1]), [1, 1]),
        (([1, 4, 7], [2, 3, 8, 9]), [1, 2, 3, 4, 7, 8, 9]),
        (([0, 5, 10], [1, 6, 11]), [0, 1, 5, 6, 10, 11]),
    ],
    reference_solution=_REF_SORTED_MERGE,
))


# ===========================================================================
# ARRAY ALGORITHMS (9 benchmarks)
# ===========================================================================

_reg(ComplexBenchmark(
    name="binary_search",
    category="array_algo",
    description="Find the index of target in a sorted array; return -1 if not found.",
    fn_name="binary_search",
    signature="def binary_search(arr: list, target: int) -> int",
    examples=[
        (([1, 3, 5, 7, 9], 5), 2),
        (([1, 3, 5, 7, 9], 1), 0),
        (([1, 3, 5, 7, 9], 9), 4),
        (([1, 3, 5, 7, 9], 4), -1),
        (([10], 10), 0),
        (([10], 5), -1),
        (([2, 4, 6, 8, 10, 12], 8), 3),
        (([1, 2, 3, 4, 5, 6, 7, 8], 6), 5),
    ],
    reference_solution=_REF_BINARY_SEARCH,
))

_reg(ComplexBenchmark(
    name="rotate_array",
    category="array_algo",
    description="Rotate an array left by k positions (first k elements move to the end).",
    fn_name="rotate_array",
    signature="def rotate_array(arr: list, k: int) -> list",
    examples=[
        (([1, 2, 3, 4, 5], 2), [3, 4, 5, 1, 2]),
        (([1, 2, 3, 4, 5], 0), [1, 2, 3, 4, 5]),
        (([1, 2, 3], 3), [1, 2, 3]),
        (([1, 2, 3, 4, 5], 5), [1, 2, 3, 4, 5]),
        (([7, 8, 9], 1), [8, 9, 7]),
        (([1], 100), [1]),
        (([1, 2, 3, 4, 5, 6], 4), [5, 6, 1, 2, 3, 4]),
    ],
    reference_solution=_REF_ROTATE_ARRAY,
))

_reg(ComplexBenchmark(
    name="longest_increasing",
    category="array_algo",
    description="Length of the longest strictly increasing subsequence.",
    fn_name="longest_increasing",
    signature="def longest_increasing(arr: list) -> int",
    examples=[
        (([10, 9, 2, 5, 3, 7, 101, 18],), 4),
        (([0, 1, 0, 3, 2, 3],), 4),
        (([7, 7, 7, 7],), 1),
        (([1],), 1),
        (([],), 0),
        (([1, 2, 3, 4, 5],), 5),
        (([5, 4, 3, 2, 1],), 1),
        (([3, 10, 2, 1, 20],), 3),
    ],
    reference_solution=_REF_LONGEST_INCREASING,
))

_reg(ComplexBenchmark(
    name="max_subarray",
    category="array_algo",
    description="Maximum sum of any contiguous subarray (Kadane's algorithm).",
    fn_name="max_subarray",
    signature="def max_subarray(arr: list) -> int",
    examples=[
        (([-2, 1, -3, 4, -1, 2, 1, -5, 4],), 6),
        (([1],), 1),
        (([5, 4, -1, 7, 8],), 23),
        (([-1, -2, -3],), -1),
        (([0, 0, 0],), 0),
        (([2, 3, -6, 4, 2, -1],), 6),
        (([1, 2, 3, 4, 5],), 15),
    ],
    reference_solution=_REF_MAX_SUBARRAY,
))

_reg(ComplexBenchmark(
    name="first_duplicate",
    category="array_algo",
    description="Return the first element that appears more than once in the array; return -1 if none.",
    fn_name="first_duplicate",
    signature="def first_duplicate(arr: list) -> int",
    examples=[
        (([2, 1, 3, 5, 3, 2],), 3),
        (([1, 2, 3, 4, 5],), -1),
        (([1, 1],), 1),
        (([5, 5, 5],), 5),
        (([],), -1),
        (([3, 0, 3, 4],), 3),
        (([7, 4, 2, 9, 4, 7],), 4),
    ],
    reference_solution=_REF_FIRST_DUPLICATE,
))

_reg(ComplexBenchmark(
    name="count_inversions",
    category="array_algo",
    description="Count the number of inversions: pairs (i, j) where i < j and arr[i] > arr[j].",
    fn_name="count_inversions",
    signature="def count_inversions(arr: list) -> int",
    examples=[
        (([2, 4, 1, 3, 5],), 3),
        (([1, 2, 3, 4],), 0),
        (([4, 3, 2, 1],), 6),
        (([1],), 0),
        (([],), 0),
        (([3, 1, 2],), 2),
        (([5, 2, 6, 1],), 4),
    ],
    reference_solution=_REF_COUNT_INVERSIONS,
))

_reg(ComplexBenchmark(
    name="two_sum_indices",
    category="array_algo",
    description="Find indices [i, j] of two numbers that add to target (i < j). Return [-1, -1] if none.",
    fn_name="two_sum_indices",
    signature="def two_sum_indices(arr: list, target: int) -> list",
    examples=[
        (([2, 7, 11, 15], 9), [0, 1]),
        (([3, 2, 4], 6), [1, 2]),
        (([3, 3], 6), [0, 1]),
        (([1, 2, 3, 4, 5], 9), [3, 4]),
        (([1, 2, 3], 7), [-1, -1]),
        (([0, 4, 3, 0], 0), [0, 3]),
        (([1, 5, 3, 7, 9], 10), [1, 4]),
    ],
    reference_solution=_REF_TWO_SUM,
))

_reg(ComplexBenchmark(
    name="flatten_chunks",
    category="array_algo",
    description="Flatten a list of chunks (given as flat array + chunk sizes) into one array.",
    fn_name="flatten_chunks",
    signature="def flatten_chunks(flat: list, chunk_sizes: list) -> list",
    examples=[
        (([1, 2, 3, 4, 5], [2, 3]), [1, 2, 3, 4, 5]),
        (([1, 2, 3, 4, 5, 6], [1, 2, 3]), [1, 2, 3, 4, 5, 6]),
        (([10, 20], [1, 1]), [10, 20]),
        (([5], [1]), [5]),
        (([], []), []),
        (([1, 2, 3, 4, 5, 6, 7], [3, 2, 2]), [1, 2, 3, 4, 5, 6, 7]),
        (([9, 8, 7, 6], [4]), [9, 8, 7, 6]),
    ],
    reference_solution=_REF_FLATTEN_CHUNKS,
))

_reg(ComplexBenchmark(
    name="sliding_window_max",
    category="array_algo",
    description="Return the maximum value in each sliding window of size k.",
    fn_name="sliding_window_max",
    signature="def sliding_window_max(arr: list, k: int) -> list",
    examples=[
        (([1, 3, -1, -3, 5, 3, 6, 7], 3), [3, 3, 5, 5, 6, 7]),
        (([1, 2, 3, 4, 5], 1), [1, 2, 3, 4, 5]),
        (([5, 4, 3, 2, 1], 5), [5]),
        (([1], 1), [1]),
        (([2, 2, 2], 2), [2, 2]),
        (([4, 3, 2, 1, 5], 3), [4, 3, 5]),
        (([1, 3, 2, 5, 4], 2), [3, 3, 5, 5]),
    ],
    reference_solution=_REF_SLIDING_WINDOW_MAX,
))


# ===========================================================================
# STRING ALGORITHMS (8 benchmarks)
# ===========================================================================

_reg(ComplexBenchmark(
    name="run_length_encode",
    category="string_algo",
    description="Run-length encode a string: 'aabbc' becomes 'a2b2c1'.",
    fn_name="run_length_encode",
    signature="def run_length_encode(s: str) -> str",
    examples=[
        (("aabbc",), "a2b2c1"),
        (("aaaa",), "a4"),
        (("abcd",), "a1b1c1d1"),
        (("",), ""),
        (("a",), "a1"),
        (("aaabbbccc",), "a3b3c3"),
        (("aabbccdd",), "a2b2c2d2"),
    ],
    reference_solution=_REF_RUN_LENGTH_ENCODE,
))

_reg(ComplexBenchmark(
    name="run_length_decode",
    category="string_algo",
    description="Decode a run-length encoded string: 'a2b2c1' becomes 'aabbc'.",
    fn_name="run_length_decode",
    signature="def run_length_decode(s: str) -> str",
    examples=[
        (("a2b2c1",), "aabbc"),
        (("a4",), "aaaa"),
        (("a1b1c1d1",), "abcd"),
        (("",), ""),
        (("z3",), "zzz"),
        (("a3b3c3",), "aaabbbccc"),
        (("x10",), "x" * 10),
    ],
    reference_solution=_REF_RUN_LENGTH_DECODE,
))

_reg(ComplexBenchmark(
    name="count_words",
    category="string_algo",
    description="Count the number of whitespace-separated words in a string.",
    fn_name="count_words",
    signature="def count_words(s: str) -> int",
    examples=[
        (("hello world",), 2),
        (("one two three",), 3),
        (("",), 0),
        (("single",), 1),
        (("  spaces   between  ",), 2),
        (("a b c d e",), 5),
        (("the quick brown fox",), 4),
    ],
    reference_solution=_REF_COUNT_WORDS,
))

_reg(ComplexBenchmark(
    name="reverse_words",
    category="string_algo",
    description="Reverse the order of words in a string (words are space-separated).",
    fn_name="reverse_words",
    signature="def reverse_words(s: str) -> str",
    examples=[
        (("hello world",), "world hello"),
        (("one two three",), "three two one"),
        (("single",), "single"),
        (("a b c",), "c b a"),
        (("the quick brown fox",), "fox brown quick the"),
        (("reverse this string please",), "please string this reverse"),
        (("go",), "go"),
    ],
    reference_solution=_REF_REVERSE_WORDS,
))

_reg(ComplexBenchmark(
    name="caesar_cipher",
    category="string_algo",
    description="Shift each alphabetic character by k positions (wrapping, preserve case). Non-alpha chars unchanged.",
    fn_name="caesar_cipher",
    signature="def caesar_cipher(s: str, k: int) -> str",
    examples=[
        (("abc", 1), "bcd"),
        (("xyz", 3), "abc"),
        (("Hello World", 13), "Uryyb Jbeyq"),
        (("ABC", 0), "ABC"),
        (("a1b2c3", 1), "b1c2d3"),
        (("ZYX", 1), "ABC"),
        (("hello", 26), "hello"),
    ],
    reference_solution=_REF_CAESAR_CIPHER,
))

_reg(ComplexBenchmark(
    name="is_palindrome_str",
    category="string_algo",
    description="Return 1 if the string is a palindrome ignoring spaces and case, 0 otherwise.",
    fn_name="is_palindrome_str",
    signature="def is_palindrome_str(s: str) -> int",
    examples=[
        (("racecar",), 1),
        (("hello",), 0),
        (("A man a plan a canal Panama",), 1),
        (("Was it a car or a cat I saw",), 1),
        (("",), 1),
        (("ab",), 0),
        (("Able was I ere I saw Elba",), 1),
    ],
    reference_solution=_REF_IS_PALINDROME_STR,
))

_reg(ComplexBenchmark(
    name="longest_common_prefix",
    category="string_algo",
    description="Find the longest common prefix of all space-separated words.",
    fn_name="longest_common_prefix",
    signature="def longest_common_prefix(s: str) -> str",
    examples=[
        (("flower flow flight",), "fl"),
        (("dog racecar car",), ""),
        (("interview inter internal",), "inter"),
        (("alone",), "alone"),
        (("abc abc abc",), "abc"),
        (("ab abc abcd",), "ab"),
        (("prefix prefix prefix",), "prefix"),
    ],
    reference_solution=_REF_LONGEST_COMMON_PREFIX,
))

_reg(ComplexBenchmark(
    name="compress_string",
    category="string_algo",
    description="Remove consecutive duplicate characters from a string.",
    fn_name="compress_string",
    signature="def compress_string(s: str) -> str",
    examples=[
        (("aabbc",), "abc"),
        (("aaabbbccc",), "abc"),
        (("abcde",), "abcde"),
        (("",), ""),
        (("aaa",), "a"),
        (("aabbaab",), "ababab"),
        (("mississippi",), "misisipi"),
    ],
    reference_solution=_REF_COMPRESS_STRING,
))


# ===========================================================================
# NUMBER THEORY (6 benchmarks)
# ===========================================================================

_reg(ComplexBenchmark(
    name="sieve_count",
    category="number_theory",
    description="Count the number of primes up to and including n (Sieve of Eratosthenes).",
    fn_name="sieve_count",
    signature="def sieve_count(n: int) -> int",
    examples=[
        ((10,), 4),
        ((1,), 0),
        ((2,), 1),
        ((20,), 8),
        ((0,), 0),
        ((50,), 15),
        ((100,), 25),
    ],
    reference_solution=_REF_SIEVE_COUNT,
))

_reg(ComplexBenchmark(
    name="count_divisors",
    category="number_theory",
    description="Count the number of positive integer divisors of n.",
    fn_name="count_divisors",
    signature="def count_divisors(n: int) -> int",
    examples=[
        ((1,), 1),
        ((6,), 4),
        ((12,), 6),
        ((7,), 2),
        ((36,), 9),
        ((100,), 9),
        ((4,), 3),
    ],
    reference_solution=_REF_COUNT_DIVISORS,
))

_reg(ComplexBenchmark(
    name="prime_factors_sum",
    category="number_theory",
    description="Sum of distinct prime factors of n.",
    fn_name="prime_factors_sum",
    signature="def prime_factors_sum(n: int) -> int",
    examples=[
        ((12,), 5),   # 2+3
        ((1,), 0),
        ((7,), 7),
        ((100,), 7),  # 2+5
        ((30,), 10),  # 2+3+5
        ((2,), 2),
        ((36,), 5),   # 2+3
    ],
    reference_solution=_REF_PRIME_FACTORS_SUM,
))

_reg(ComplexBenchmark(
    name="modular_pow",
    category="number_theory",
    description="Compute a^b mod m using fast exponentiation. Return 1 % m when b=0.",
    fn_name="modular_pow",
    signature="def modular_pow(a: int, b: int, m: int) -> int",
    examples=[
        ((2, 10, 1000), 24),
        ((3, 0, 5), 1),
        ((2, 0, 1), 0),
        ((5, 3, 13), 8),
        ((7, 4, 11), 9),
        ((2, 8, 256), 0),
        ((3, 5, 7), 5),
    ],
    reference_solution=_REF_MODULAR_POW,
))

_reg(ComplexBenchmark(
    name="digital_root",
    category="number_theory",
    description="Repeatedly sum digits of n until a single digit remains.",
    fn_name="digital_root",
    signature="def digital_root(n: int) -> int",
    examples=[
        ((493,), 7),
        ((0,), 0),
        ((9,), 9),
        ((999,), 9),
        ((1,), 1),
        ((101,), 2),
        ((9875,), 2),
    ],
    reference_solution=_REF_DIGITAL_ROOT,
))

_reg(ComplexBenchmark(
    name="catalan",
    category="number_theory",
    description="Compute the nth Catalan number (C(0)=1, C(1)=1, C(2)=2, C(3)=5, ...).",
    fn_name="catalan",
    signature="def catalan(n: int) -> int",
    examples=[
        ((0,), 1),
        ((1,), 1),
        ((2,), 2),
        ((3,), 5),
        ((4,), 14),
        ((5,), 42),
        ((6,), 132),
    ],
    reference_solution=_REF_CATALAN,
))


# ===========================================================================
# DYNAMIC PROGRAMMING (5 benchmarks)
# ===========================================================================

_reg(ComplexBenchmark(
    name="coin_change_count",
    category="dp",
    description="Count the number of ways to make the given amount using coins [1, 2, 5].",
    fn_name="coin_change_count",
    signature="def coin_change_count(amount: int) -> int",
    examples=[
        ((5,), 4),
        ((0,), 1),
        ((1,), 1),
        ((2,), 2),
        ((10,), 10),
        ((3,), 2),
        ((7,), 6),
    ],
    reference_solution=_REF_COIN_CHANGE_COUNT,
))

_reg(ComplexBenchmark(
    name="climb_stairs",
    category="dp",
    description="Count ways to climb n stairs taking 1 or 2 steps at a time.",
    fn_name="climb_stairs",
    signature="def climb_stairs(n: int) -> int",
    examples=[
        ((1,), 1),
        ((2,), 2),
        ((3,), 3),
        ((4,), 5),
        ((5,), 8),
        ((0,), 1),
        ((10,), 89),
    ],
    reference_solution=_REF_CLIMB_STAIRS,
))

_reg(ComplexBenchmark(
    name="min_coin_change",
    category="dp",
    description="Minimum number of coins [1, 2, 5] to make amount. Return -1 if impossible.",
    fn_name="min_coin_change",
    signature="def min_coin_change(amount: int) -> int",
    examples=[
        ((11,), 3),
        ((0,), 0),
        ((1,), 1),
        ((5,), 1),
        ((6,), 2),
        ((3,), 2),
        ((10,), 2),
    ],
    reference_solution=_REF_MIN_COIN_CHANGE,
))

_reg(ComplexBenchmark(
    name="house_robber",
    category="dp",
    description="Max sum of non-adjacent elements in an array (house robber problem).",
    fn_name="house_robber",
    signature="def house_robber(arr: list) -> int",
    examples=[
        (([1, 2, 3, 1],), 4),
        (([2, 7, 9, 3, 1],), 12),
        (([],), 0),
        (([5],), 5),
        (([1, 1],), 1),
        (([2, 1, 1, 2],), 4),
        (([10, 5, 10],), 20),
    ],
    reference_solution=_REF_HOUSE_ROBBER,
))

_reg(ComplexBenchmark(
    name="triangle_path_sum",
    category="dp",
    description=(
        "Minimum path sum from top to bottom of a triangle encoded as a flat array "
        "(row 0 has 1 element, row 1 has 2, etc.)."
    ),
    fn_name="triangle_path_sum",
    signature="def triangle_path_sum(flat: list) -> int",
    examples=[
        (([2, 3, 4, 6, 5, 7, 4, 1, 8, 3],), 11),
        (([1],), 1),
        (([1, 2, 3],), 3),
        (([3, 7, 4, 2, 4, 6, 8, 5, 9, 3],), 11),
        (([2],), 2),
        (([1, 2, 3, 4, 5, 6],), 7),
        (([5, 8, 6, 4, 7, 3],), 12),
    ],
    reference_solution=_REF_TRIANGLE_PATH,
))


# ===========================================================================
# GRAPH (2 benchmarks)
# ===========================================================================

_reg(ComplexBenchmark(
    name="bfs_distance",
    category="graph_encoded",
    description=(
        "Shortest path distance (BFS) in an undirected unweighted graph. "
        "Input encoded as [n_nodes, n_edges, u1, v1, u2, v2, ..., src, dst]. "
        "Return -1 if no path exists."
    ),
    fn_name="bfs_distance",
    signature="def bfs_distance(encoded: list) -> int",
    examples=[
        # 4 nodes, 4 edges: 0-1, 1-2, 2-3, 0-3; src=0, dst=3
        (([4, 4, 0, 1, 1, 2, 2, 3, 0, 3, 0, 3],), 1),
        # 3 nodes, 2 edges: 0-1, 1-2; src=0, dst=2
        (([3, 2, 0, 1, 1, 2, 0, 2],), 2),
        # src == dst
        (([3, 2, 0, 1, 1, 2, 1, 1],), 0),
        # Disconnected: 3 nodes, 1 edge: 0-1; src=0, dst=2
        (([3, 1, 0, 1, 0, 2],), -1),
        # Linear: 5 nodes 0-1-2-3-4; src=0, dst=4
        (([5, 4, 0, 1, 1, 2, 2, 3, 3, 4, 0, 4],), 4),
        # 2 nodes, 1 edge: 0-1; src=1, dst=0
        (([2, 1, 0, 1, 1, 0],), 1),
        # Star: node 0 connected to 1,2,3,4; src=1, dst=3
        (([5, 4, 0, 1, 0, 2, 0, 3, 0, 4, 1, 3],), 2),
    ],
    reference_solution=_REF_BFS_DISTANCE,
))

_reg(ComplexBenchmark(
    name="connected_count",
    category="graph_encoded",
    description=(
        "Count connected components in an undirected graph. "
        "Input encoded as [n_nodes, n_edges, u1, v1, ...]."
    ),
    fn_name="connected_count",
    signature="def connected_count(encoded: list) -> int",
    examples=[
        # 4 nodes, 2 edges: 0-1, 2-3 → 2 components
        (([4, 2, 0, 1, 2, 3],), 2),
        # 3 nodes, 0 edges → 3 components
        (([3, 0],), 3),
        # 3 nodes, 3 edges: 0-1, 1-2, 0-2 → 1 component
        (([3, 3, 0, 1, 1, 2, 0, 2],), 1),
        # 1 node → 1 component
        (([1, 0],), 1),
        # 5 nodes, 2 edges: 0-1, 3-4 → 3 components
        (([5, 2, 0, 1, 3, 4],), 3),
        # 6 nodes, 4 edges: 0-1, 0-2, 3-4, 3-5 → 2 components
        (([6, 4, 0, 1, 0, 2, 3, 4, 3, 5],), 2),
        # All connected: 4 nodes, 3 edges: chain 0-1-2-3 → 1 component
        (([4, 3, 0, 1, 1, 2, 2, 3],), 1),
    ],
    reference_solution=_REF_CONNECTED_COUNT,
))


# ===========================================================================
# GRAPH (3 additional benchmarks → total 5 in graph_encoded)
# ===========================================================================

_reg(ComplexBenchmark(
    name="graph_degree_sum",
    category="graph_encoded",
    description=(
        "Sum of all node degrees in an undirected graph. "
        "Input encoded as [n_nodes, n_edges, u1, v1, ...]. "
        "For an undirected graph this equals 2 * n_edges."
    ),
    fn_name="graph_degree_sum",
    signature="def graph_degree_sum(encoded: list) -> int",
    examples=[
        # 3 nodes, 2 edges: 0-1, 1-2 → degrees [1,2,1] → sum=4
        (([3, 2, 0, 1, 1, 2],), 4),
        # 4 nodes, 0 edges → sum=0
        (([4, 0],), 0),
        # 2 nodes, 1 edge: 0-1 → sum=2
        (([2, 1, 0, 1],), 2),
        # 4 nodes, 4 edges: triangle 0-1-2-0 + 0-3 → sum=8
        (([4, 4, 0, 1, 1, 2, 2, 0, 0, 3],), 8),
        # 5 nodes, 4 edges: star from 0 → degrees [4,1,1,1,1] → sum=8
        (([5, 4, 0, 1, 0, 2, 0, 3, 0, 4],), 8),
        # 1 node, 0 edges → sum=0
        (([1, 0],), 0),
        # 3 nodes, 3 edges: complete triangle → sum=6
        (([3, 3, 0, 1, 1, 2, 0, 2],), 6),
    ],
    reference_solution=_REF_GRAPH_DEGREE_SUM,
))

_reg(ComplexBenchmark(
    name="graph_max_degree",
    category="graph_encoded",
    description=(
        "Maximum degree of any node in an undirected graph. "
        "Input encoded as [n_nodes, n_edges, u1, v1, ...]."
    ),
    fn_name="graph_max_degree",
    signature="def graph_max_degree(encoded: list) -> int",
    examples=[
        # 3 nodes, 2 edges: 0-1, 1-2 → degrees [1,2,1] → max=2
        (([3, 2, 0, 1, 1, 2],), 2),
        # 4 nodes, 0 edges → max=0
        (([4, 0],), 0),
        # 5 nodes star from 0 → max=4
        (([5, 4, 0, 1, 0, 2, 0, 3, 0, 4],), 4),
        # Complete triangle → max=2
        (([3, 3, 0, 1, 1, 2, 0, 2],), 2),
        # 2 nodes, 1 edge → max=1
        (([2, 1, 0, 1],), 1),
        # 4 nodes chain 0-1-2-3 → max=2 (node 1 or 2)
        (([4, 3, 0, 1, 1, 2, 2, 3],), 2),
        # 1 node, 0 edges → max=0
        (([1, 0],), 0),
    ],
    reference_solution=_REF_GRAPH_MAX_DEGREE,
))

_reg(ComplexBenchmark(
    name="graph_has_cycle",
    category="graph_encoded",
    description=(
        "Return 1 if the undirected graph has a cycle, 0 otherwise. "
        "Input encoded as [n_nodes, n_edges, u1, v1, ...]."
    ),
    fn_name="graph_has_cycle",
    signature="def graph_has_cycle(encoded: list) -> int",
    examples=[
        # Triangle: 3 nodes, 3 edges → cycle
        (([3, 3, 0, 1, 1, 2, 0, 2],), 1),
        # Tree: 3 nodes, 2 edges (chain) → no cycle
        (([3, 2, 0, 1, 1, 2],), 0),
        # 4 nodes, 0 edges → no cycle
        (([4, 0],), 0),
        # 4 nodes, 4 edges: 0-1-2-3-0 square → cycle
        (([4, 4, 0, 1, 1, 2, 2, 3, 3, 0],), 1),
        # 4 nodes, 3 edges: spanning tree → no cycle
        (([4, 3, 0, 1, 0, 2, 0, 3],), 0),
        # 2 nodes, 1 edge → no cycle
        (([2, 1, 0, 1],), 0),
        # 5 nodes, 5 edges: tree + extra edge → cycle
        (([5, 5, 0, 1, 1, 2, 2, 3, 3, 4, 0, 4],), 1),
    ],
    reference_solution=_REF_GRAPH_HAS_CYCLE,
))


# ===========================================================================
# ARRAY ALGORITHMS (3 additional benchmarks)
# ===========================================================================

_reg(ComplexBenchmark(
    name="product_except_self",
    category="array_algo",
    description="Return array where each element is the product of all other elements (no division).",
    fn_name="product_except_self",
    signature="def product_except_self(arr: list) -> list",
    examples=[
        (([1, 2, 3, 4],), [24, 12, 8, 6]),
        (([2, 3, 4],), [12, 8, 6]),
        (([1, 1, 1, 1],), [1, 1, 1, 1]),
        (([5],), [1]),
        (([2, 2, 2, 2, 2],), [16, 16, 16, 16, 16]),
        (([1, 2, 3],), [6, 3, 2]),
        (([0, 1, 2, 3],), [6, 0, 0, 0]),
    ],
    reference_solution=_REF_PRODUCT_EXCEPT_SELF,
))

_reg(ComplexBenchmark(
    name="find_missing",
    category="array_algo",
    description="Array contains n-1 distinct integers from 1..n. Find the missing one.",
    fn_name="find_missing",
    signature="def find_missing(arr: list) -> int",
    examples=[
        (([1, 2, 4, 5, 6],), 3),
        (([2, 3, 4, 5],), 1),
        (([1, 2, 3, 4],), 5),
        (([1],), 2),
        (([2],), 1),
        (([1, 2, 3, 5, 6, 7],), 4),
        (([1, 2, 4],), 3),
    ],
    reference_solution=_REF_FIND_MISSING,
))

_reg(ComplexBenchmark(
    name="majority_element",
    category="array_algo",
    description="Find the element that appears more than n/2 times (guaranteed to exist).",
    fn_name="majority_element",
    signature="def majority_element(arr: list) -> int",
    examples=[
        (([3, 2, 3],), 3),
        (([2, 2, 1, 1, 1, 2, 2],), 2),
        (([1],), 1),
        (([5, 5, 5, 1, 5],), 5),
        (([3, 3, 4, 2, 4, 4, 2, 4, 4],), 4),
        (([7, 7, 7],), 7),
        (([1, 2, 1, 2, 1],), 1),
    ],
    reference_solution=_REF_MAJORITY_ELEMENT,
))


# ===========================================================================
# NUMBER THEORY (2 additional benchmarks)
# ===========================================================================

_reg(ComplexBenchmark(
    name="is_perfect",
    category="number_theory",
    description="Return 1 if n is a perfect number (sum of proper divisors equals n), else 0.",
    fn_name="is_perfect",
    signature="def is_perfect(n: int) -> int",
    examples=[
        ((6,), 1),   # 1+2+3=6
        ((28,), 1),  # 1+2+4+7+14=28
        ((1,), 0),
        ((12,), 0),
        ((496,), 1),
        ((2,), 0),
        ((100,), 0),
    ],
    reference_solution=_REF_IS_PERFECT,
))

_reg(ComplexBenchmark(
    name="gcd",
    category="number_theory",
    description="Compute the greatest common divisor of two non-negative integers.",
    fn_name="gcd",
    signature="def gcd(a: int, b: int) -> int",
    examples=[
        ((48, 18), 6),
        ((0, 5), 5),
        ((5, 0), 5),
        ((12, 8), 4),
        ((7, 13), 1),
        ((100, 75), 25),
        ((1, 1), 1),
    ],
    reference_solution=_REF_GCD,
))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_benchmarks_by_category(category: str) -> list[ComplexBenchmark]:
    """Return all benchmarks matching the given category string."""
    return [b for b in COMPLEX_BENCHMARK_REGISTRY if b.category == category]


def get_all_categories() -> list[str]:
    """Return sorted list of unique category names."""
    return sorted(set(b.category for b in COMPLEX_BENCHMARK_REGISTRY))
