# A Codex of Fundamental Algorithms: Python Implementations and Performance Analysis

## Introduction

This report serves as a definitive codex of fundamental algorithms for the modern computer science practitioner. It provides not only canonical Python implementations but also a nuanced analysis of their performance characteristics and the design trade-offs inherent in their various forms. The focus is on bridging theoretical principles with practical, high-quality code. The document is organized into thematic sections, from foundational number theory and searching to advanced graph traversal and dynamic programming paradigms. Each algorithm is presented with a concise explanation, a formal complexity analysis, and illustrative code to facilitate a deep and functional understanding.

## Section 1: Core Algorithmic Paradigms

Algorithms can be classified by the design strategy or paradigm they employ. Understanding these paradigms is crucial as they provide a high-level framework for approaching and solving new computational problems.

### 1.1 Brute Force

The brute-force paradigm is the most straightforward approach to problem-solving. It involves systematically enumerating all possible candidates for the solution and checking whether each candidate satisfies the problem's statement. While often not the most efficient, it is a valuable starting point for understanding a problem's complexity and can be effective for small problem sizes.

A classic example is the Traveling Salesman Problem (TSP), which seeks the shortest possible route that visits a set of cities and returns to the origin city. A brute-force solution involves calculating the length of every possible permutation of cities and selecting the shortest one.

```python
import itertools
import math

def calculate_total_distance(route, dist_matrix):
    """Calculates the total distance of a given route."""
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += dist_matrix[route[i]][route[i+1]]
    # Add distance from the last city back to the start
    total_distance += dist_matrix[route[-1]][route[0]] # Fixed: route[0] for returning to start
    return total_distance

def tsp_brute_force(dist_matrix):
    """
    Solves the Traveling Salesman Problem using a brute-force approach.
    dist_matrix[i][j] is the distance from city i to city j.
    """
    num_cities = len(dist_matrix)
    if num_cities == 0:
        return None, 0
        
    cities = list(range(num_cities))
    # Generate all possible routes (permutations) starting from city 0
    all_routes = list(itertools.permutations(cities[1:])) # Permutations of all cities except the start
    
    min_distance = float('inf')
    best_route = None
    
    for route_suffix in all_routes:
        # Complete route starts and ends at city 0
        current_route = [0] + list(route_suffix) # Fixed: added [0] at the start
        current_distance = calculate_total_distance(current_route, dist_matrix)
        
        if current_distance < min_distance:
            min_distance = current_distance
            best_route = current_route
            
    return best_route, min_distance

# Example usage:
# dist_matrix = [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]]
# route, distance = tsp_brute_force(dist_matrix)
# print(f"Best route: {route}, Distance: {distance}")

```

**Time Complexity:** $O(n!)$
The number of possible routes grows factorially with the number of cities, making this approach impractical for all but the smallest instances of the problem.

### 1.2 Recursion

Recursion is a powerful programming technique where a function calls itself to solve smaller, self-similar instances of the same problem. A recursive function must have one or more base cases that terminate the recursion, preventing infinite loops. This approach often leads to elegant and intuitive solutions that mirror the mathematical definition of a problem.

Many algorithms covered in this report, such as the naive Fibonacci implementation, Depth-First Search, Merge Sort, and Quick Sort, are naturally expressed using recursion.

### 1.3 Divide and Conquer

Divide and Conquer is a recursive strategy that breaks a problem into smaller, independent subproblems of the same type, solves them recursively, and then combines their solutions to solve the original problem. This paradigm is the foundation for many of the most efficient algorithms in computer science.

Key examples detailed later in this report include:

* Merge Sort and Quick Sort, which divide an array into smaller subarrays to be sorted.

* Binary Search, which repeatedly divides the search space in half.

### 1.4 Greedy Algorithms

Greedy algorithms build a solution step-by-step by making the locally optimal choice at each stage, with the hope of finding a global optimum. While this strategy does not guarantee an optimal solution for all problems, it is highly effective and efficient for certain classes of problems, such as finding minimum spanning trees or shortest paths with non-negative weights.

#### 1.4.1 Prim's Algorithm

Prim's algorithm finds a Minimum Spanning Tree (MST) for a connected, undirected graph. An MST is a subset of edges that connects all vertices with the minimum possible total weight. The algorithm is greedy because it grows the MST by adding the cheapest edge that connects a vertex in the MST to a vertex outside the MST.

```python
import heapq

def prims_algorithm(graph):
    """
    Finds the Minimum Spanning Tree using Prim's algorithm.
    Graph is an adjacency list: {'A': [('B', 1), ('C', 4)], 'B': [('A', 1), ('C', 2)], ...}
    """
    if not graph:
        return [], 0

    start_node = list(graph.keys())[0] # Pick an arbitrary start node
    mst = []
    visited = {start_node}
    # Priority queue stores (weight, start_node, end_node)
    edges = [(weight, start_node, neighbor) for neighbor, weight in graph[start_node]]
    heapq.heapify(edges)
    
    total_weight = 0

    while edges and len(visited) < len(graph):
        weight, u, v = heapq.heappop(edges)
        if v not in visited:
            visited.add(v)
            mst.append((u, v, weight))
            total_weight += weight
            for next_neighbor, next_weight in graph[v]:
                if next_neighbor not in visited:
                    heapq.heappush(edges, (next_weight, v, next_neighbor))
    
    return mst, total_weight

```

**Time Complexity:** $O((V+E)\log V)$ with a priority queue, where $V$ is vertices and $E$ is edges.

#### 1.4.2 Kruskal's Algorithm

Kruskal's algorithm also finds an MST but uses a different greedy strategy. It sorts all edges in the graph by weight in non-decreasing order and adds edges to the MST as long as they do not form a cycle with the edges already added. To efficiently detect cycles, a Union-Find data structure is used.

```python
def kruskals_algorithm(graph_edges, num_vertices):
    """
    Finds the Minimum Spanning Tree using Kruskal's algorithm.
    graph_edges is a list of tuples: [(weight, u, v),...]
    """
    parent = list(range(num_vertices))
    def find_set(v):
        if v == parent[v]:
            return v
        parent[v] = find_set(parent[v])
        return parent[v]

    def union_sets(a, b):
        a = find_set(a)
        b = find_set(b)
        if a != b:
            parent[b] = a
            return True
        return False

    mst = []
    total_weight = 0
    graph_edges.sort() # Sort edges by weight

    for weight, u, v in graph_edges:
        if union_sets(u, v): # If they are not already in the same set (i.e., adding this edge won't form a cycle)
            mst.append((u, v, weight))
            total_weight += weight
            if len(mst) == num_vertices - 1: # MST has V-1 edges
                break
    
    return mst, total_weight

```

**Time Complexity:** $O(E \log E)$ dominated by sorting the edges.

#### 1.4.3 Huffman Coding

Huffman Coding is a greedy algorithm used for lossless data compression. It assigns variable-length codes to characters based on their frequencies: more frequent characters get shorter codes, and less frequent characters get longer codes. The algorithm builds a binary tree (Huffman Tree) from the bottom up, repeatedly merging the two nodes with the lowest frequencies until only one root node remains.

```python
import heapq
from collections import defaultdict

def huffman_coding(text):
    """Generates Huffman codes for a given text."""
    if not text:
        return {}, None

    frequency = defaultdict(int)
    for char in text:
        frequency[char] += 1

    # Node for the Huffman Tree
    class HuffmanNode:
        def __init__(self, char, freq):
            self.char = char
            self.freq = freq
            self.left = None
            self.right = None
        
        def __lt__(self, other):
            return self.freq < other.freq

    # Build priority queue (min-heap)
    priority_queue = [HuffmanNode(char, freq) for char, freq in frequency.items()]
    heapq.heapify(priority_queue)

    # Build the Huffman Tree
    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(priority_queue, merged)

    huffman_tree_root = priority_queue[0] # The last remaining node is the root
    huffman_codes = {}

    # Traverse the tree to generate codes
    def generate_codes(node, current_code):
        if node is None:
            return
        if node.char is not None: # It's a leaf node
            huffman_codes[node.char] = current_code
            return
        generate_codes(node.left, current_code + "0")
        generate_codes(node.right, current_code + "1")

    generate_codes(huffman_tree_root, "")
    return huffman_codes, huffman_tree_root

```

### 1.5 Dynamic Programming

Dynamic Programming (DP) is a powerful algorithmic technique for solving optimization and counting problems by breaking them down into simpler, overlapping subproblems. It solves each subproblem only once and stores its solution, typically in a table (or array), thereby avoiding redundant computations. This section explores two classic problems that are elegantly solved using DP: the 0/1 Knapsack problem and the Longest Common Subsequence problem.

#### 1.5.1 The 0/1 Knapsack Problem

The 0/1 Knapsack problem is a cornerstone of combinatorial optimization. Given a set of items, each with a specific weight and value, the goal is to determine which items to include in a knapsack of a fixed capacity to maximize the total value. In the "0/1" variant, the decision for each item is binary: either take the entire item or leave it behind.

The dynamic programming solution involves constructing a 2D table, `dp[i][w]`, where `dp[i][w]` represents the maximum value that can be achieved using a subset of the first `i` items with a total weight capacity of `w`. The table is filled based on the following recurrence relation: for each item `i` and capacity `w`, the optimal choice is the maximum of either (1) not including item `i`, or (2) including item `i` (if its weight does not exceed the current capacity).

```python
def knapsack_01(values, weights, capacity):
    """
    Solves the 0/1 Knapsack problem using dynamic programming.
    
    :param values: A list of item values.
    :param weights: A list of item weights.
    :param capacity: The maximum weight capacity of the knapsack.
    :return: The maximum value that can be achieved.
    """
    n = len(values)
    # dp[i][w] will be the maximum value that can be put in a knapsack of
    # capacity w using the first i items.
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            # The item's index is i-1 because of the 0-based list index.
            item_weight = weights[i-1]
            item_value = values[i-1]
            
            if item_weight <= w:
                # Choice: include the item or not include the item.
                dp[i][w] = max(
                    item_value + dp[i-1][w - item_weight],  # Include item i
                    dp[i-1][w]                               # Don't include item i
                )
            else:
                # The current item is too heavy, so we can't include it.
                dp[i][w] = dp[i-1][w]
                
    return dp[n][capacity]

```

**Time Complexity:** $O(nW)$
Where $n$ is the number of items and $W$ is the knapsack capacity. The complexity is determined by the size of the DP table, as we must fill each of the $n \times W$ cells.

**Space Complexity:** $O(nW)$
The space is required for the $n \times W$ DP table. This can be optimized to $O(W)$ by realizing that to compute the current row, only the previous row is needed.

#### 1.5.2 The Longest Common Subsequence (LCS) Problem

The Longest Common Subsequence (LCS) problem involves finding the longest subsequence that is present in two given sequences. A subsequence is derived from another sequence by deleting some or no elements without changing the order of the remaining elements. For example, "ACE" is a subsequence of "ABCDE".

The DP solution constructs a 2D table, `L[i][j]`, which stores the length of the LCS for the prefixes of the two sequences: $X[0...i-1]$ and $Y[0...j-1]$. The recurrence relation is as follows:

* If the characters at the current positions match ($X[i-1] == Y[j-1]$), the LCS length is one greater than the LCS of the preceding prefixes: $L[i][j] = 1 + L[i-1][j-1]$.

* If the characters do not match, the LCS length is the maximum of the LCS lengths found by excluding one character from either sequence: $L[i][j] = \max(L[i-1][j], L[i][j-1])$.

```python
def longest_common_subsequence(s1, s2):
    """
    Finds the length of the Longest Common Subsequence of two strings
    using dynamic programming.
    """
    m, n = len(s1), len(s2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = 1 + dp[i-1][j-1]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # The length of the LCS is in the bottom-right cell
    return dp[m][n]
    
    # To reconstruct the LCS string, one would backtrack from dp[m][n].
    # This part is omitted for brevity but follows the arrows in the conceptual model.

```

**Time Complexity:** $O(mn)$
Where $m$ and $n$ are the lengths of the two sequences. The complexity is determined by the need to fill every cell of the $m \times n$ DP table.

**Space Complexity:** $O(mn)$
The space is required for the DP table. Similar to the knapsack problem, this can be optimized to $O(\min(m,n))$ if only the length is required.

### 1.6 Backtracking

Backtracking is an algorithmic technique for solving problems recursively by trying to build a solution incrementally, one piece at a time, and removing those solutions that fail to satisfy the constraints of the problem at any point in time. It systematically explores the solution space, and if a partial solution cannot be completed, it "backtracks" by undoing the last choice and trying an alternative.

#### 1.6.1 N-Queens Problem

The N-Queens puzzle is the problem of placing N chess queens on an N×N chessboard so that no two queens threaten each other. The backtracking algorithm places queens column by column, and for each column, it tries to place a queen in a row where it is not attacked by already placed queens. If a valid row is found, it recursively moves to the next column. If no valid row is found in the current column, it backtracks to the previous column and moves the queen there to a new row.

```python
def solve_n_queens(n):
    """Solves the N-Queens problem using backtracking."""
    solutions = []
    board = [['.' for _ in range(n)] for _ in range(n)]

    def is_safe(row, col):
        # Check this row on left side
        for i in range(col):
            if board[row][i] == 'Q':
                return False
        # Check upper diagonal on left side
        for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
            if board[i][j] == 'Q':
                return False
        # Check lower diagonal on left side
        for i, j in zip(range(row, n, 1), range(col, -1, -1)):
            if board[i][j] == 'Q':
                return False
        return True

    def solve(col):
        if col >= n:
            solutions.append(["".join(row) for row in board])
            return
        
        for i in range(n):
            if is_safe(i, col):
                board[i][col] = 'Q'
                solve(col + 1)
                board[i][col] = '.' # Backtrack

    solve(0)
    return solutions
```
**Time Complexity:** $O(N!)$ in the worst-case, as it explores a permutation-like search space, though pruning makes it much faster in practice.

#### 1.6.2 Sudoku Solver

Sudoku can be solved efficiently using backtracking. The algorithm iterates through empty cells, trying to place a valid number (1-9) in each. A number is valid if it doesn't already exist in the current row, column, or 3x3 subgrid. If a valid number is placed, the algorithm recursively moves to the next empty cell. If no valid number can be placed, it backtracks to the previous cell, erases the number, and tries the next one.

```python
def solve_sudoku(board):
    """Solves a Sudoku puzzle using backtracking."""
    
    def find_empty(bo):
        for i in range(len(bo)):
            for j in range(len(bo[0])): # Fixed: len(bo[0]) for column length
                if bo[i][j] == 0:
                    return (i, j)  # row, col
        return None

    def is_valid(bo, num, pos):
        # Check row
        for i in range(len(bo[0])): # Fixed: len(bo[0]) for column length
            if bo[pos[0]][i] == num and pos[1] != i: # Fixed: pos[0] for row, pos[1] for col
                return False
        # Check column
        for i in range(len(bo)):
            if bo[i][pos[1]] == num and pos[0] != i: # Fixed: pos[1] for col, pos[0] for row
                return False
        # Check 3x3 box
        box_x = pos[1] // 3 # Fixed: pos[1] for col
        box_y = pos[0] // 3 # Fixed: pos[0] for row
        for i in range(box_y*3, box_y*3 + 3):
            for j in range(box_x * 3, box_x*3 + 3):
                if bo[i][j] == num and (i,j) != pos:
                    return False
        return True

    def solve():
        find = find_empty(board)
        if not find:
            return True
        else:
            row, col = find

        for i in range(1, 10):
            if is_valid(board, i, (row, col)):
                board[row][col] = i
                if solve():
                    return True
                board[row][col] = 0 # Backtrack
        return False

    solve()
    return board
```

### 1.7 Randomized Algorithms

Randomized algorithms use random numbers to make decisions during their execution. This can help avoid worst-case scenarios that might arise from specific data patterns, often leading to better average-case performance.

#### 1.7.1 Randomized Quick Sort

A prime example is Randomized Quick Sort. Standard Quick Sort's performance degrades to $O(n^2)$ if the pivot selection is consistently poor (e.g., on an already sorted array). By choosing the pivot randomly, the algorithm ensures that no specific input can reliably trigger this worst-case behavior. The expected time complexity becomes $O(n \log n)$ for any input.

```python
import random

def randomized_quick_sort(arr):
    """
    Sorts an array using Randomized Quick Sort.
    This is not an in-place implementation for simplicity.
    """
    if len(arr) <= 1:
        return arr
    else:
        # Choose a random pivot
        pivot_index = random.randint(0, len(arr) - 1)
        pivot = arr[pivot_index]
        
        # Create a new list without the pivot
        remaining_elements = arr[:pivot_index] + arr[pivot_index+1:]
        
        left = [x for x in remaining_elements if x <= pivot]
        right = [x for x in remaining_elements if x > pivot]
        
        return randomized_quick_sort(left) + [pivot] + randomized_quick_sort(right)
```
**Time Complexity:** Average: $\Theta(n \log n)$, Worst: $O(n^2)$
While the worst-case complexity remains $O(n^2)$, randomization makes it extremely unlikely to occur.

## Section 2: Foundational Algorithms by Problem Type

This section organizes algorithms based on the type of problem they are designed to solve, covering searching, sorting, graph problems, and more.

### 2.1 Searching Algorithms

Searching algorithms are designed to find a specific element within a data structure.

#### 2.1.1 Linear Search

Linear search is the most basic search algorithm. It sequentially checks each element of a list until a match is found or the whole list has been searched. Its main advantage is its simplicity and the fact that it does not require the data to be sorted.

```python
def linear_search(arr, target):
    """
    Performs a linear search for a target in an array.
    Returns the index of the target, or -1 if not found.
    """
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
```
**Time Complexity:** $O(n)$
In the worst case, the algorithm must check every element in the list.

**Space Complexity:** $O(1)$ for the iterative version.

#### 2.1.2 Binary Search

Binary search is a highly efficient algorithm for finding a specific element within a sorted array. Its effectiveness comes from a "divide and conquer" strategy: it repeatedly divides the search interval in half until the target element is found or the interval becomes empty. The prerequisite that the data must be sorted is critical for the algorithm's correctness and performance.

**Implementation 1: Iterative Approach**
The iterative implementation uses a `while` loop with two pointers, `left` and `right`, to define the boundaries of the current search interval. At each step, it calculates the middle index, compares the element at that index with the target, and narrows the interval accordingly. This approach is often preferred in production environments because it avoids the overhead of recursion and has a constant space complexity.

```python
def binary_search_iterative(arr, target):
    """
    Performs binary search for a target value in a sorted array using an iterative approach.
    Returns the index of the target, or -1 if not found.
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        # Using (left + right) // 2 can lead to overflow in some languages for large indices.
        # This alternative is safer, though less of a concern in Python.
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
            
    return -1
```
**Time Complexity:** $O(\log n)$
With each comparison, the algorithm eliminates half of the remaining search space. The number of steps required to narrow down from $n$ elements to one is logarithmic.

**Space Complexity:** $O(1)$
The iterative version uses a constant amount of extra space for the `left`, `right`, and `mid` pointers.

**Implementation 2: Recursive Approach**
The recursive implementation mirrors the divide-and-conquer logic more directly. The function calls itself on the appropriate half of the array until the base case (element found or subarray is empty) is reached. While conceptually elegant, this approach consumes memory on the function call stack.

```python
def binary_search_recursive(arr, left, right, target):
    """
    Performs binary search using a recursive approach.
    """
    if left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            return binary_search_recursive(arr, mid + 1, right, target)
        else:
            return binary_search_recursive(arr, left, mid - 1, target)
            
    return -1
```
**Time Complexity:** $O(\log n)$
The time complexity remains logarithmic, as the problem size is halved at each recursive step.

**Space Complexity:** $O(\log n)$
Each recursive call adds a frame to the call stack. In the worst case, the depth of the recursion is proportional to $\log n$.

**Implementation 3: Using Python's `bisect` Module**
Python's standard library includes the `bisect` module, which provides highly optimized and robust functions for binary search operations. These functions are primarily used to find the correct insertion point for an element to maintain the sorted order of a list, but they can be adapted for searching.

```python
import bisect

def binary_search_bisect(arr, target):
    """
    Performs binary search using Python's built-in bisect module.
    """
    # bisect_left finds the insertion point for target, which is the index
    # of the first element equal to or greater than target.
    index = bisect.bisect_left(arr, target)
    
    # Check if the found index is valid and if the element at that index is the target.
    if index < len(arr) and arr[index] == target:
        return index
    else:
        return -1
```

### 2.2 Sorting Algorithms

Sorting algorithms arrange elements in a specific order. They are a fundamental part of computer science, with a wide range of algorithms developed for different scenarios.

#### 2.2.1 Simple Sorting Algorithms

These algorithms are typically easy to understand and implement but are inefficient for large datasets, often having a time complexity of $O(n^2)$.

**Bubble Sort**
Bubble Sort repeatedly steps through the list, compares adjacent elements, and swaps them if they are in the wrong order. The pass through the list is repeated until the list is sorted. Smaller elements "bubble" to the top of the list.

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
```
**Time Complexity:** $O(n^2)$ in the average and worst cases.

**Insertion Sort**
Insertion Sort builds the final sorted array one item at a time. It iterates through the input elements and inserts each element into its correct position in the sorted part of the array. It is efficient for small datasets or nearly sorted arrays.

```python
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr
```
**Time Complexity:** $O(n^2)$ in the average and worst cases, but $O(n)$ in the best case (already sorted array).

**Selection Sort**
Selection Sort divides the input list into a sorted and an unsorted region. It repeatedly selects the smallest (or largest) element from the unsorted region and moves it to the end of the sorted region.

```python
def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
```
**Time Complexity:** $O(n^2)$ in all cases (best, average, and worst).

#### 2.2.2 Efficient Sorting Algorithms

These algorithms offer significantly better performance for large datasets, typically with an average time complexity of $O(n \log n)$.

**Merge Sort**
Merge Sort is a quintessential "divide and conquer" algorithm. It operates by recursively splitting the input list into two halves until each sublist contains a single element (which is inherently sorted). It then repeatedly merges these sorted sublists back together, producing larger sorted lists until the entire list is sorted.

The core of the algorithm lies in the `merge` helper function, which takes two sorted sublists and combines them into a single sorted list in linear time.

```python
def merge_sort(arr):
    """
    Sorts an array using the Merge Sort algorithm.
    """
    if len(arr) <= 1:
        return arr

    # Divide the array into two halves
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]

    # Recursively sort both halves
    sorted_left = merge_sort(left_half)
    sorted_right = merge_sort(right_half)

    # Merge the sorted halves
    return merge(sorted_left, sorted_right)

def merge(left, right):
    """
    Merges two sorted lists into a single sorted list.
    """
    result = []
    i, j = 0, 0

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Append any remaining elements
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result
```
**Time Complexity:** $\Theta(n \log n)$
This complexity holds for the best, average, and worst cases. The $\log n$ factor arises from the number of times the array is recursively divided in half. The $n$ factor comes from the work performed at each level of recursion by the `merge` function, which must iterate through all elements to combine the sublists.

**Space Complexity:** $O(n)$
The algorithm requires additional space proportional to the input size to hold the temporary sublists during the merging process.

**Properties:** Merge Sort is a stable sort, meaning that the relative order of equal elements is preserved in the sorted output. It is not an in-place sort due to its space requirements.

**Quick Sort**
Quick Sort is another powerful divide-and-conquer algorithm that is often faster in practice than other $O(n \log n)$ algorithms due to smaller constant factors and cache-friendly behavior. It works by selecting a 'pivot' element from the array and partitioning the other elements into two sub-arrays based on whether they are less than or greater than the pivot. The pivot is placed in its final sorted position. This process is then applied recursively to the sub-arrays.

The performance of Quick Sort is highly dependent on the choice of pivot. A good pivot splits the array into roughly equal-sized partitions, leading to balanced recursion. A poor pivot leads to unbalanced partitions and degrades performance. The following implementation uses the Lomuto partition scheme, a common and straightforward approach that typically selects the last element as the pivot.

```python
def quick_sort_lomuto(arr, low, high):
    """
    Sorts an array using the Quick Sort algorithm with Lomuto partition scheme.
    This is an in-place implementation.
    """
    if low < high:
        # pi is the partitioning index, arr[pi] is now at the right place
        pi = partition_lomuto(arr, low, high)
        
        # Separately sort elements before and after partition
        quick_sort_lomuto(arr, low, pi - 1)
        quick_sort_lomuto(arr, pi + 1, high)

def partition_lomuto(arr, low, high):
    """
    This function takes the last element as pivot, places the pivot element at its
    correct position in the sorted array, and places all smaller elements to the left
    of the pivot and all greater elements to the right.
    """
    pivot = arr[high]
    i = low - 1  # Index of smaller element

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
```
**Time Complexity:** Average: $\Theta(n \log n)$, Worst: $O(n^2)$
The average-case complexity is excellent. However, the worst-case complexity of $O(n^2)$ occurs when the pivot selection consistently produces highly unbalanced partitions. For example, if the last element is chosen as the pivot and the array is already sorted, each partition will divide the array into a sub-array of size 0 and a sub-array of size $n-1$, leading to a linear recursion depth.

**Space Complexity:** $O(\log n)$
For the in-place implementation, the space complexity is determined by the depth of the recursion stack, which is $O(\log n)$ on average for balanced partitions.

**Properties:** Quick Sort is an unstable sort. The in-place version is highly memory-efficient.

**Heap Sort**
Heap Sort is an efficient in-place sorting algorithm that leverages the properties of a binary heap data structure. The algorithm consists of two main phases. First, it transforms the input array into a max-heap, a specialized binary tree where the value of each parent node is greater than or equal to the values of its children. This ensures the largest element is at the root of the heap. Second, it repeatedly extracts the maximum element (the root), swaps it with the last element in the unsorted portion of the array, reduces the heap size by one, and restores the max-heap property for the remaining elements.

```python
def heap_sort(arr):
    """
    Sorts an array using the Heap Sort algorithm.
    """
    n = len(arr)

    # Build a max-heap from the array.
    # The last parent node is at index (n // 2) - 1.
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # One by one extract elements from the heap.
    for i in range(n - 1, 0, -1):
        # Move current root (max element) to the end.
        arr[i], arr[0] = arr[0], arr[i]
        # Call max heapify on the reduced heap.
        heapify(arr, i, 0)

def heapify(arr, n, i):
    """
    To heapify a subtree rooted at index i. n is the size of the heap.
    """
    largest = i      # Initialize largest as root
    left = 2 * i + 1   # Left child
    right = 2 * i + 2  # Right child

    # See if left child of root exists and is greater than root
    if left < n and arr[left] > arr[largest]:
        largest = left

    # See if right child of root exists and is greater than largest so far
    if right < n and arr[right] > arr[largest]:
        largest = right

    # Change root if needed
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # Swap
        # Heapify the root of the affected subtree.
        heapify(arr, n, largest)
```
**Time Complexity:** $\Theta(n \log n)$
This complexity holds for all cases. Building the initial max-heap takes $O(n)$ time. Subsequently, the algorithm performs $n-1$ swaps and heapify operations. Each heapify operation on the shrinking heap takes $O(\log n)$ time, leading to an overall complexity of $O(n \log n)$.

**Space Complexity:** $O(1)$
Heap Sort is an in-place algorithm, meaning it sorts the array without requiring significant additional storage that scales with the input size. The `heapify` function can be implemented recursively, which would use $O(\log n)$ stack space, but it is often implemented iteratively for a true $O(1)$ space solution.

**Properties:** Heap Sort is an unstable sort. Its main advantages are its guaranteed worst-case time complexity and its in-place nature.

#### 2.2.3 Comparative Analysis of Sorting Algorithms

The choice of a sorting algorithm is not arbitrary but a critical design decision based on the specific constraints and requirements of the application. There is no single "best" algorithm; each presents a unique profile of trade-offs between time efficiency, memory usage, and stability. The following table provides a comparative summary to aid in this decision-making process.

| Algorithm      | Time Complexity (Average) | Time Complexity (Worst) | Space Complexity (Worst) | Stable | In-Place | Paradigm          |
| :------------- | :------------------------ | :---------------------- | :----------------------- | :----- | :------- | :---------------- |
| Merge Sort     | $\Theta(n \log n)$        | $\Theta(n \log n)$      | $O(n)$                   | Yes    | No       | Divide and Conquer |
| Quick Sort     | $\Theta(n \log n)$        | $O(n^2)$                | $O(\log n)$              | No     | Yes      | Divide and Conquer |
| Heap Sort      | $\Theta(n \log n)$        | $\Theta(n \log n)$      | $O(1)$                   | No     | Yes      | Heap-based        |

This comparison reveals the nuanced landscape of sorting. Quick Sort is frequently the fastest in practice for general-purpose sorting due to its low constant factors and efficient cache utilization, but its $O(n^2)$ worst-case performance is a significant liability in mission-critical systems where predictability is paramount. Merge Sort offers the guarantee of $\Theta(n \log n)$ performance and stability, making it ideal for scenarios where the relative order of equal elements must be maintained, but this reliability comes at the cost of $O(n)$ auxiliary space. Heap Sort provides a compelling alternative, delivering the same guaranteed $\Theta(n \log n)$ time complexity as Merge Sort but with the $O(1)$ space efficiency of an in-place algorithm. However, it is typically slower than a well-implemented Quick Sort and is not stable.

### 2.3 Graph Algorithms

Graph algorithms operate on data structures composed of vertices (nodes) and edges (connections), which are used to model networks and relationships.

#### 2.3.1 Graph Traversal

Graph traversal is the process of visiting (checking and/or updating) each vertex in a graph. Such traversals are fundamental to solving a vast array of problems in computer science, from network routing to artificial intelligence. The two most essential traversal algorithms are Breadth-First Search (BFS) and Depth-First Search (DFS).

**Breadth-First Search (BFS)**
Breadth-First Search is a graph traversal algorithm that explores vertices "level by level." It begins at a specified source node, visits all of its immediate neighbors, then visits the neighbors of those neighbors, and so on. This systematic, expanding-wave exploration ensures that it discovers all nodes at a given distance (in terms of number of edges) before moving to nodes at the next distance level. A key property of BFS is that it is guaranteed to find the shortest path from the source to any other node in an unweighted graph.

The behavior of BFS is dictated by its use of a queue, a First-In, First-Out (FIFO) data structure. The queue ensures that nodes are processed in the order they are discovered, which is the mechanism that drives the level-by-level traversal.

```python
from collections import deque, defaultdict

def bfs(graph, start_node):
    """
    Performs Breadth-First Search on a graph starting from a given node.
    The graph is represented as an adjacency list (dictionary).
    Returns the list of visited nodes in BFS order.
    """
    if start_node not in graph:
        return []

    visited = set()
    queue = deque([start_node])
    visited.add(start_node)
    
    traversal_order = []

    while queue:
        # Dequeue a vertex from the front of the queue
        vertex = queue.popleft()
        traversal_order.append(vertex)

        # Enqueue all adjacent, unvisited vertices
        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                
    return traversal_order
```
**Time Complexity:** $O(V+E)$
Where $V$ is the number of vertices and $E$ is the number of edges. Each vertex is enqueued and dequeued exactly once, and every edge is examined once when its source vertex is dequeued. Thus, the complexity is linear in the size of the graph representation.

**Space Complexity:** $O(V)$
In the worst-case scenario, the queue may need to hold all vertices at a single level. For a wide, shallow graph (like a star graph), this can be up to $O(V)$ vertices. The visited set also contributes $O(V)$ space.

**Depth-First Search (DFS)**
Depth-First Search explores a graph by traversing as far as possible along each branch before backtracking. It prioritizes going "deep" into the graph structure first. When it reaches a node with no unvisited neighbors, it backtracks to the previous node and explores the next available path.

The behavior of DFS is governed by its use of a stack, a Last-In, First-Out (LIFO) data structure. The stack ensures that the most recently discovered node is the next one to be explored. This can be implemented using an explicit stack or, more commonly, through recursion, which uses the system's call stack implicitly.

**Recursive Approach**
The recursive implementation is often considered more elegant and intuitive. The function maintains a set of visited nodes and calls itself for each unvisited neighbor of the current node.

```python
def dfs_recursive(graph, start_node, visited=None):
    """
    Performs Depth-First Search on a graph using recursion.
    Returns the list of visited nodes in DFS order.
    """
    if visited is None:
        visited = set()
    
    traversal_order = []
    if start_node not in visited:
        visited.add(start_node)
        traversal_order.append(start_node)
        for neighbor in graph.get(start_node, []):
            traversal_order.extend(dfs_recursive(graph, neighbor, visited))
            
    return traversal_order
```

**Iterative Approach**
The iterative approach uses an explicit stack to manage the nodes to be visited. This method avoids potential recursion depth limits in Python and can be more memory-efficient for very deep or pathological graphs.

```python
def dfs_iterative(graph, start_node):
    """
    Performs Depth-First Search on a graph using an explicit stack.
    """
    if start_node not in graph:
        return []
        
    visited = set()
    stack = [start_node]
    traversal_order = []

    while stack:
        vertex = stack.pop()
        
        if vertex not in visited:
            visited.add(vertex)
            traversal_order.append(vertex)
            
            # Add neighbors to the stack. Note the reverse order to mimic
            # the recursive version's typical exploration order.
            for neighbor in reversed(graph.get(vertex, [])):
                if neighbor not in visited:
                    stack.append(neighbor)
                    
    return traversal_order
```
**Time Complexity:** $O(V+E)$
Similar to BFS, each vertex is pushed onto the stack and popped exactly once, and every edge is examined once.

**Space Complexity:** $O(V)$
In the worst case, for a graph that is a long, unbranching chain, the stack (either explicit or the call stack) could hold all $V$ vertices simultaneously.

#### 2.3.2 Shortest Path Algorithms

Finding the shortest path between nodes in a graph is a classic and critical problem with applications in network routing, logistics, and mapping.

**Dijkstra's Algorithm**
Dijkstra's algorithm is a greedy algorithm that solves the single-source shortest path problem for a weighted graph, provided that all edge weights are non-negative. It systematically finds the shortest path from a given source node to every other node in the graph.

The algorithm maintains a set of visited nodes and a data structure that stores the tentative shortest distance from the source to every other node. It operates as follows:
1.  Initialize the distance to the source node as 0 and all other distances as infinity.
2.  Maintain a priority queue of unvisited nodes, prioritized by their tentative distance.
3.  While the priority queue is not empty:
    a.  Extract the node `u` with the smallest known distance from the queue. This distance is now considered final.
    b.  For each neighbor `v` of `u`, perform a "relaxation" step: calculate the distance to `v` by passing through `u` (i.e., `distance(u) + weight(u, v)`). If this new path is shorter than the currently known distance to `v`, update `v`'s distance.

Once a node is extracted from the priority queue, it is marked as visited, and its shortest path is finalized.

The efficiency of Dijkstra's algorithm hinges on its use of a min-priority queue (often implemented with a min-heap), which allows for the efficient retrieval of the unvisited node with the smallest tentative distance at each step.

```python
import heapq

def dijkstra(graph, start_node):
    """
    Finds the shortest path from a start node to all other nodes in a weighted graph
    using Dijkstra's algorithm.
    Graph is an adjacency list: {'node': {'neighbor': weight,...},...}
    Returns a dictionary of shortest distances from the start node.
    """
    # Initialize distances with infinity, except for the start node
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    
    # Priority queue to store (distance, node) tuples
    priority_queue = [(0, start_node)]
    
    while priority_queue:
        # Get the node with the smallest distance
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # If we've found a shorter path already, skip
        if current_distance > distances[current_node]:
            continue
            
        # Explore neighbors
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            
            # If a shorter path to the neighbor is found
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
                
    return distances
```
**Time Complexity:** $O((E+V)\log V)$
With a binary heap-based priority queue, the complexity is dominated by the priority queue operations. Every vertex is added to the queue once. For every edge, there might be a distance update, which involves a `heappush` operation taking $O(\log V)$ time. Therefore, the total time is a combination of vertex extractions and edge relaxations.

**Space Complexity:** $O(V)$
The space is required to store the distances dictionary and the priority queue, which in the worst case can hold all vertices.

**Floyd-Warshall Algorithm**
The Floyd-Warshall algorithm is a dynamic programming algorithm that finds the shortest paths between all pairs of vertices in a weighted graph. It can handle graphs with positive or negative edge weights, but it cannot handle negative-weight cycles. It works by iteratively considering each vertex as an intermediate point in the paths between all other pairs of vertices and updating the path if a shorter one is found.

```python
def floyd_warshall(graph):
    """
    Finds all-pairs shortest paths using the Floyd-Warshall algorithm.
    Graph is an adjacency matrix. Use float('inf') for no direct path.
    """
    V = len(graph)
    dist = list(map(lambda i: list(map(lambda j: j, i)), graph))

    for k in range(V):
        for i in range(V):
            for j in range(V):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    return dist
```
**Time Complexity:** $O(V^3)$
The algorithm's complexity comes from its three nested loops, which iterate through all vertices.

**Space Complexity:** $O(V^2)$
It requires a matrix to store the distances between all pairs of vertices.

#### 2.3.3 Topological Sort

Topological sorting provides a linear ordering of vertices in a Directed Acyclic Graph (DAG) such that for every directed edge from vertex `u` to vertex `v`, `u` comes before `v` in the ordering. It is commonly used for scheduling tasks with dependencies.

**Kahn's Algorithm**
Kahn's algorithm is a popular method for topological sorting. It works by repeatedly finding nodes with an in-degree of 0 (no incoming edges), adding them to the sorted list, and then "removing" them and their outgoing edges from the graph. This process continues until no nodes with an in-degree of 0 are left.

```python
from collections import deque, defaultdict

def topological_sort_kahn(num_vertices, edges):
    """
    Performs topological sort using Kahn's algorithm.
    edges is a list of pairs (u, v).
    """
    adj = defaultdict(list)
    in_degree = [0] * num_vertices
    for u, v in edges:
        adj[u].append(v)
        in_degree[v] += 1

    queue = deque([i for i in range(num_vertices) if in_degree[i] == 0])
    top_order = []

    while queue:
        u = queue.popleft()
        top_order.append(u)

        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(top_order) == num_vertices:
        return top_order
    else:
        return "Graph contains a cycle" # Topological sort not possible
```

### 2.4 String Algorithms

String algorithms are designed for processing and manipulating sequences of characters.

#### 2.4.1 Pattern Searching: Knuth-Morris-Pratt (KMP)

The Knuth-Morris-Pratt (KMP) algorithm is a highly efficient string-matching algorithm. It searches for occurrences of a "pattern" within a "text" by using information from previous matches to avoid re-examining characters. Its key innovation is a precomputed array, often called the Longest Proper Prefix (LPS) array, which stores the length of the longest proper prefix of the pattern that is also a suffix.

```python
def kmp_search(text, pattern):
    """
    Finds all occurrences of a pattern in text using the KMP algorithm.
    """
    def compute_lps_array(pat):
        m = len(pat)
        lps = [0] * m
        length = 0  # Length of the previous longest prefix suffix
        i = 1
        while i < m:
            if pat[i] == pat[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps

    n = len(text)
    m = len(pattern)
    lps = compute_lps_array(pattern)
    i = 0  # index for text
    j = 0  # index for pattern
    occurrences = []

    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1
        
        if j == m:
            occurrences.append(i - j)
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return occurrences
```

### 2.5 Hashing Algorithms

Hashing is the process of converting an input of arbitrary size into a fixed-size value, known as a hash value or hash code. This is done using a hash function. Hashing is fundamental to the implementation of hash tables.

#### 2.5.1 Hashing and Hash Tables

A hash table is a data structure that maps keys to values for highly efficient lookup. In Python, the built-in `dict` and `set` types are implemented using hash tables. A hash function is used to compute an index into an array of "buckets" or "slots," from which the desired value can be found. A good hash function distributes keys uniformly across the buckets to minimize collisions—when two different keys hash to the same index.

A common method for handling collisions is chaining, where each bucket is a list (or another data structure) that stores all key-value pairs that hash to that index.

```python
class SimpleHashTable:
    """A simple hash table implementation using chaining."""
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(self.size)]

    def _hash_function(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        index = self._hash_function(key)
        bucket = self.table[index]
        # Update value if key already exists
        for i, pair in enumerate(bucket):
            if pair[0] == key:
                bucket[i] = (key, value)
                return
        # Otherwise, add new key-value pair
        bucket.append((key, value))

    def search(self, key):
        index = self._hash_function(key)
        bucket = self.table[index]
        for k, v in bucket:
            if k == key:
                return v
        return None # Key not found
```

### 2.6 Mathematical and Number Theory Algorithms

This category includes algorithms designed to solve mathematical problems or operate on numbers.

#### 2.6.1 The Fibonacci Sequence

The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones, typically starting with 0 and 1. It is defined by the recurrence relation $F(n)=F(n-1)+F(n-2)$, with seed values $F(0)=0$ and $F(1)=1$. This sequence is a classic introductory problem in computer science because it elegantly illustrates the concepts of recursion, iterative processes, and the optimization paradigm of dynamic programming.

**Implementation 1: Naive Recursive Approach**
A direct translation of the mathematical recurrence into a Python function yields a simple and readable implementation. This approach is valued for its clarity and close resemblance to the formal definition of the sequence.

```python
def fibonacci_recursive(n):
    """
    Calculates the nth Fibonacci number using a naive recursive approach.
    """
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)
```
**Time Complexity:** $O(2^n)$
The time complexity is exponential because the function branches into two recursive calls for each value of $n$ greater than 1. This creates a recursion tree where the same subproblems (e.g., `fibonacci_recursive(3)`) are computed multiple times across different branches, leading to a massive number of redundant calculations.

**Space Complexity:** $O(n)$
The space complexity is linear, determined by the maximum depth of the recursion call stack. To calculate $F(n)$, the stack depth will be proportional to $n$.

**Implementation 2: Iterative (Bottom-Up) Approach**
A more performant method involves building the sequence from the beginning, or "bottom-up." This iterative approach uses a loop and a constant number of variables to store the two preceding values needed to calculate the next term in the sequence. It completely avoids the overhead and redundancy of recursion.

```python
def fibonacci_iterative(n):
    """
    Calculates the nth Fibonacci number using an iterative (bottom-up) approach.
    """
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b
```
**Time Complexity:** $O(n)$
The algorithm performs a single pass through a loop that runs approximately $n$ times. Each iteration involves a fixed number of arithmetic operations, resulting in a linear time complexity.

**Space Complexity:** $O(1)$
This method uses a fixed number of variables (`a`, `b`, and the loop counter) regardless of the size of $n$. This constant space usage makes it highly memory-efficient.

**Implementation 3: Optimized Recursive Approach with Memoization**
This technique, also known as top-down dynamic programming, combines the declarative elegance of recursion with the efficiency of the iterative approach. It works by caching the results of subproblems. When the function is called with a given input, it first checks if the result has already been computed and stored. If so, it returns the cached result; otherwise, it computes the result, stores it in the cache, and then returns it. This ensures that each Fibonacci number is calculated only once. In Python, this can be implemented cleanly using the `functools.lru_cache` decorator.

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_memoized(n):
    """
    Calculates the nth Fibonacci number using a memoized recursive approach.
    The @lru_cache decorator handles the caching automatically.
    """
    if n <= 1:
        return n
    return fibonacci_memoized(n - 1) + fibonacci_memoized(n - 2)
```
**Time Complexity:** $O(n)$
Although the function is recursive, the cache ensures that each subproblem from `fibonacci_memoized(0)` to `fibonacci_memoized(n)` is computed only once. Subsequent calls are resolved in $O(1)$ time from the cache.

**Space Complexity:** $O(n)$
The space is required for both the recursion call stack and the cache, both of which grow linearly with $n$.

#### 2.6.2 Prime Generation: The Sieve of Eratosthenes

The Sieve of Eratosthenes is an ancient and remarkably efficient algorithm for finding all prime numbers up to a specified integer, $n$. Instead of testing each number for primality individually, it operates by iteratively marking as composite the multiples of each prime it discovers.

The algorithm proceeds as follows:
1.  Create a boolean list, `is_prime`, of size $n+1$ and initialize all its entries to `True`. Mark the entries for 0 and 1 as `False`, as they are not prime.
2.  Iterate from $p=2$ up to $\sqrt{n}$. The iteration only needs to go up to $\sqrt{n}$ because if a number $k$ has a factor larger than its square root, it must also have one smaller than it.
3.  If `is_prime[p]` remains `True`, it means $p$ is a prime number. Then, iterate through the multiples of $p$ (starting from $p^2$) and mark them as `False`.

After the loops complete, any index `i` for which `is_prime[i]` is `True` corresponds to a prime number.

```python
def sieve_of_eratosthenes(n):
    """
    Generates all prime numbers up to n using the Sieve of Eratosthenes.
    Returns a list of primes.
    """
    if n < 2:
        return []
    
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for p in range(2, int(n**0.5) + 1):
        if is_prime[p]:
            # Mark all multiples of p from p*p onwards.
            # Multiples smaller than p*p would have been marked by smaller primes.
            for multiple in range(p * p, n + 1, p):
                is_prime[multiple] = False
                
    primes = [i for i, prime in enumerate(is_prime) if prime]
    return primes
```
**Time Complexity:** $O(n \log(\log n))$
This complexity arises from the sum of operations for marking multiples. For each prime $p$, we perform approximately $n/p$ marking operations. The sum over all primes up to $n$ ($n/2+n/3+n/5+...$) converges to $O(n \log(\log n))$, a result related to the harmonic series of prime reciprocals. This is significantly faster than the $O(n\sqrt{n})$ complexity of naively testing each number for primality.

**Space Complexity:** $O(n)$
The algorithm requires a boolean array of size $n+1$ to store the primality status of each number, resulting in linear space complexity.

#### 2.6.3 Euclidean Algorithm

The Euclidean algorithm is an efficient method for computing the greatest common divisor (GCD) of two integers, which is the largest number that divides them both without leaving a remainder.

```python
def gcd_euclidean(a, b):
    """
    Computes the greatest common divisor (GCD) of a and b
    using the Euclidean algorithm.
    """
    while b:
        a, b = b, a % b
    return a
```

#### 2.6.4 Exponentiation by Squaring

Exponentiation by squaring, also known as binary exponentiation, is a fast method for calculating large integer powers of a number. It significantly reduces the number of multiplications required compared to repeated multiplication. The algorithm is based on the observation that for a power $n$, if $n$ is even, then $x^n=(x^2)^{n/2}$, and if $n$ is odd, then $x^n=x \cdot x^{n-1}$.

```python
def power_by_squaring(base, exp):
    """
    Calculates base^exp using exponentiation by squaring.
    """
    res = 1
    # base %= 1000000007 # Example modulus - removed as it's an example
    while exp > 0:
        # If exponent is odd, multiply base with result
        if exp % 2 == 1:
            # res = (res * base) % 1000000007 # Example modulus - removed
            res = res * base
        # Exponent must be even now
        # base = (base * base) % 1000000007 # Example modulus - removed
        base = base * base
        exp //= 2
    return res
```
**Time Complexity:** $O(\log n)$, where $n$ is the exponent, as the exponent is halved in each step.

## Section 3: Application: Expression Parsing and Evaluation

A classic computer science problem that integrates data structures and algorithmic logic is the evaluation of mathematical expressions from a string. This task requires careful handling of operator precedence and nested structures like parentheses.

### 3.1 Basic Calculator Expression Evaluation

The problem is to evaluate a string representing a mathematical expression, such as "(1+(4+5+2)-3)+(6+8)", respecting the standard order of operations (BODMAS/PEMDAS), without using built-in functions like `eval()`.

A robust approach for this is to use an adaptation of Dijkstra's Shunting-yard algorithm, which uses two stacks: one for numeric values and one for operators. This method correctly handles operator precedence and parentheses by converting the standard infix notation to a postfix (Reverse Polish Notation) representation, which is then trivial to evaluate.

A simplified version of this problem, common in technical interviews, involves expressions with `+`, `-`, `*`, and `/` but no parentheses. This can be solved efficiently in a single pass using one stack. The key is to process higher-precedence operators (`*`, `/`) immediately, while deferring lower-precedence ones (`+`, `-`).

```python
def evaluate_expression(s: str) -> int:
    """
    Evaluates a string expression with +, -, *, / operators and integers.
    This implementation correctly handles operator precedence. No parentheses.
    """
    if not s:
        return 0

    stack = []
    current_num = 0
    operator = '+'
    
    for i, char in enumerate(s):
        if char.isdigit():
            current_num = current_num * 10 + int(char)
        
        # Process the number when we hit an operator or the end of the string
        if not char.isdigit() and not char.isspace() or i == len(s) - 1:
            if operator == '+':
                stack.append(current_num)
            elif operator == '-':
                stack.append(-current_num)
            elif operator == '*':
                stack.append(stack.pop() * current_num)
            elif operator == '/':
                # Python's int division truncates towards negative infinity.
                # The problem often requires truncation towards zero.
                last_num = stack.pop()
                if last_num < 0:
                    stack.append(-(-last_num // current_num))
                else:
                    stack.append(last_num // current_num)

            # Update for the next number
            operator = char
            current_num = 0
            
    return sum(stack)
```

## Conclusion

This report has traversed a curated selection of fundamental algorithms, from number theory and searching to complex graph traversals and dynamic programming solutions. The analysis consistently reveals that the design and selection of an algorithm is a process of navigating critical trade-offs. There is no universally "best" algorithm; the optimal choice is always contingent upon the specific constraints of the problem at hand, including the nature of the input data, memory limitations, and the need for predictable performance.

Three major algorithmic paradigms have been explored:
* **Divide and Conquer**, exemplified by Merge Sort, Quick Sort, and Binary Search, demonstrates the power of recursively breaking down a problem into smaller, more manageable subproblems. Its efficiency is often logarithmic in nature, a direct result of this recursive partitioning.
* **Greedy Algorithms**, showcased by Dijkstra's algorithm, build a solution step-by-step by making the locally optimal choice at each stage. While often simpler and faster, their correctness depends on the problem possessing the greedy choice property, a condition that does not always hold.
* **Dynamic Programming**, illustrated by the Fibonacci sequence, Knapsack problem, and LCS, provides a systematic method for solving complex problems with overlapping subproblems. By storing the results of subproblems, it transforms otherwise intractable exponential-time problems into solvable polynomial-time ones.

A recurring theme is the profound and inseparable relationship between algorithms and data structures. The choice between a queue and a stack entirely dictates the traversal pattern of BFS versus DFS. The use of a priority queue is what makes Dijkstra's algorithm efficient. The 2D table is the very foundation of the DP solutions for Knapsack and LCS. This underscores that data structures are not mere implementation details but are core components that enable and shape an algorithm's logic and behavior. A deep, functional understanding of these foundational algorithms and their underlying principles is, therefore, indispensable for the engineering of efficient, scalable, and robust computational solutions.
