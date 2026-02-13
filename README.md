# CS4050 Lab 1: Graph Representations at Scale
![GitHub Latest Pre-Release)](https://img.shields.io/github/v/release/angrynarwhal/4050-Lab-1?include_prereleases&label=pre-release&logo=github)  

Place all of your submission documents in the [./lab1_submission_folder](./lab1_submission/). You will submit a zip file of that directory in Canvas. 

## Overview

This lab explores the **practical performance differences** between adjacency matrix and adjacency list graph representations. Rather than just learning the theoretical complexities, you will:

1. **Generate** road networks at various scales (100 to 100,000+ nodes)
2. **Implement** algorithms using both representations
3. **Measure** actual memory usage and runtime performance
4. **Experience** where each representation succeeds and fails
5. **Understand** why representation choice matters in real systems

If you are curious about the use of OO, classes, polymorphism, and overloading of methods in the pythonic example, and its corrallaries in C, you can read up on that in [This File](./polymorphism-overloading.md). 

## Learning Objectives

By completing this lab, you will:

- Understand **O(V²) vs O(V+E)** memory tradeoffs through direct experience
- See why sparse graphs (like road networks) favor adjacency lists
- Observe **O(1) vs O(degree)** edge lookup tradeoffs
- Understand why **O(V) vs O(degree)** neighbor iteration affects algorithm choice
- Develop intuition for choosing representations in real applications

## Quick Start

```bash
# 1. Generate test networks
cd scripts
python3 generate_network.py --size tiny    # 100 nodes - baseline
python3 generate_network.py --size small   # 1,000 nodes
python3 generate_network.py --size medium  # 10,000 nodes - matrix starts struggling
python3 generate_network.py --size large   # 50,000 nodes - matrix ~10GB RAM!

# 2. Run experiments
cd ../python
python3 run_experiments.py

# 3. Or run individual benchmarks
python graph_representations.py data/medium_nodes.csv data/medium_edges.csv --benchmark
```

## Network Sizes and Expected Behavior

| Size | Nodes | Matrix Memory | Expected Behavior |
|------|-------|---------------|-------------------|
| tiny | 100 | ~80 KB | No visible difference between representations |
| small | 1,000 | ~8 MB | Minimal difference, good for debugging |
| medium | 10,000 | ~800 MB | Matrix noticeably slower to load |
| large | 50,000 | ~20 GB | Matrix takes minutes to allocate, algorithms slow |
| huge | 100,000 | ~80 GB | **Matrix will crash your 32GB machine!** |

The "huge" size is intentionally designed to demonstrate the failure mode of O(V²) memory scaling.

## Repository Structure

```
cs3050-Lab-6/
├── data/                           # Generated network files
│   ├── tiny_nodes.csv
│   ├── tiny_edges.csv
│   ├── small_nodes.csv
│   └── ...
├── lab1_submission/                           # Generated network files
│   ├── .keep  # Ensures an empty directory is cloned locally
│   ├── <<Your Submission in Markdown Format (you can copy that section of the README.md to start)>>
│   └── ...
├── scripts/
│   └── generate_network.py         # Network data generator
├── python/
│   ├── graph_representations.py    # Core implementations
│   └── run_experiments.py          # Guided experiment runner
├── c/
│   └── (C implementations)
└── README.md
```

## Understanding the Code

### Graph Representations

**Adjacency List** (`AdjacencyListGraph`):
```python
# Memory: O(V + E)
# Stores: dict of node_id -> list of (neighbor_id, weight)

self.adj_list = {
    0: [(1, 2.5), (3, 1.8)],  # Node 0 connects to 1 and 3
    1: [(0, 2.5), (2, 3.0)],  # Node 1 connects to 0 and 2
    ...
}
```

**Adjacency Matrix** (`AdjacencyMatrixGraph`):
```python
# Memory: O(V²) regardless of edge count!
# Stores: 2D array where matrix[i][j] = weight if edge exists, 0 otherwise

self.matrix = [
    [0.0, 2.5, 0.0, 1.8],  # Node 0's edges
    [2.5, 0.0, 3.0, 0.0],  # Node 1's edges
    ...
]
```

### Key Operations Comparison

| Operation | Adjacency List | Adjacency Matrix |
|-----------|----------------|------------------|
| Memory | O(V + E) | O(V²) |
| Add edge | O(1) | O(1) |
| Check if edge exists | O(degree) | **O(1)** ← Matrix wins |
| Get all neighbors | **O(degree)** ← List wins | O(V) |
| Iterate all edges | O(E) | O(V²) |

### Why This Matters for Algorithms

**Dijkstra's algorithm** repeatedly calls `get_neighbors()`:
- With adjacency list: O(degree) per call → O((V+E) log V) total
- With adjacency matrix: O(V) per call → O(V² log V) total

For a road network with V=50,000 nodes and average degree 6:
- List-based Dijkstra: ~300,000 neighbor lookups
- Matrix-based Dijkstra: ~2.5 billion cell scans!

## Lab Exercises

### Exercise 1: Memory Scaling Analysis

Run experiments on progressively larger networks and record:

| Network | Nodes | List Memory | Matrix Memory | Ratio |
|---------|-------|-------------|---------------|-------|
| tiny | | | | |
| small | | | | |
| medium | | | | |
| large | | | | |

**Questions:**
1. What is the relationship between node count and matrix memory?
2. At what size does the matrix become impractical?
3. Predict the memory for 200,000 nodes. Would it fit in 32GB?

### Exercise 2: Edge Query Benchmark

The matrix has O(1) edge lookup. Measure the speedup:

```python
# Test code is in benchmark_edge_queries() # In the C program
# Run 10,000 random edge existence checks
```

**Questions:**
1. How much faster is matrix edge lookup?
2. Why is list lookup slower? Trace through the code.
3. In what applications would fast edge lookup matter?

### Exercise 3: Algorithm Performance

Run Dijkstra's algorithm on both representations:

| Network | List Dijkstra | Matrix Dijkstra | Slowdown |
|---------|---------------|-----------------|----------|
| tiny | | | |
| small | | | |
| medium | | | |

Dijkstra's algorithm is in the `c` folder, and instructions for running it are [in the README.md file in that directory.](./c/README.md)

**Questions:**
1. Why is matrix-based Dijkstra slower despite O(1) edge lookup?
2. Where in the algorithm does the slowdown occur?
3. For what graph density would matrix be faster?

### Exercise 4: The Breaking Point (Demonstration)

**WARNING: Make sure all other work on your computer is saved before initiating this!**

Try loading the "huge" network with matrix representation:
```bash
python3 scripts/generate_network.py --size huge
python3 -c "from python/graph_representations import load_graph; load_graph('data/huge_nodes.csv', 'data/huge_edges.csv', use_matrix=True)"
```

Watch your system monitor. Document:
- Memory usage before loading
- Memory usage during matrix allocation
- What happens when memory is exhausted

### Exercise 5: Real-World Decision Making

Given these scenarios, choose the appropriate representation and justify:

1. **Google Maps routing**: ~50 million road intersections, avg degree 4
2. **Social network analysis**: 1 billion users, avg 200 friends
3. **Circuit analysis**: 10,000 components, each connected to 3 others
4. **Dense communication matrix**: 500 servers, all-to-all connections

## Implementing Your Own Graph Class

As an additional challenge, implement a **hybrid representation** that:
- Uses adjacency list for sparse regions
- Uses matrix blocks for dense subgraphs
- Automatically chooses based on local density

```python
class HybridGraph(Graph):
    def __init__(self, density_threshold=0.1):
        # Your implementation here
        pass
```

## Submission Requirements

Place all of your submission documents in the [./lab1_submission_folder](./lab1_submission/). You will submit a zip file of that directory in Canvas. 

1. **Experimental data** from all exercises (CSV or screenshots)
2. **Written answers** to all questions (~1-2 paragraphs each)
3. **Analysis** of when each representation is appropriate
4. **Code** for any modifications you made
5. **Reflection** on what surprised you in the experiments

## Additional Resources

- [Dijkstra's Algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)
- [A* Search](https://en.wikipedia.org/wiki/A*_search_algorithm)
- [Graph Data Structures](https://en.wikipedia.org/wiki/Graph_(abstract_data_type))
- [SNAP Datasets](https://snap.stanford.edu/data/) - Real-world network data

## Troubleshooting

**"Matrix would require X GB"**: Working as intended! This demonstrates why O(V²) doesn't scale.

**Out of memory**: Kill the process with Ctrl+C. Use `--size medium` or smaller.

**Slow matrix allocation**: For 50,000 nodes, allocation takes 30-60 seconds. Be patient.

**Different results each run**: Shortest path benchmarks use random endpoints. Set `random.seed(42)` for reproducibility.


