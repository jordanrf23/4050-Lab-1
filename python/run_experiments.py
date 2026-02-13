#!/usr/bin/env python3
"""
CS3050 Lab 6 - Student Experiment Runner

This script guides you through experiments comparing adjacency matrix
and adjacency list representations.

Run this script and follow the prompts to:
1. Generate networks of increasing size
2. Measure memory usage and load times
3. Compare algorithm performance
4. Observe where each representation excels (and fails!)
"""

import os
import sys
import time
import csv
import tracemalloc

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_representations import (
    AdjacencyListGraph, AdjacencyMatrixGraph,
    load_graph, dijkstra, astar,
    benchmark_edge_queries, benchmark_neighbor_iteration, benchmark_shortest_path
)


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}")


def print_section(text: str):
    """Print a section header."""
    print(f"\n--- {text} ---")


def get_data_path(filename: str) -> str:
    """Get path to data file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, 'data', filename)


def check_data_files(size: str) -> tuple:
    """Check if data files exist for a given size."""
    nodes_file = get_data_path(f'{size}_nodes.csv')
    edges_file = get_data_path(f'{size}_edges.csv')
    
    nodes_exist = os.path.exists(nodes_file)
    edges_exist = os.path.exists(edges_file)
    
    return nodes_file, edges_file, nodes_exist and edges_exist


def count_nodes_in_file(filepath: str) -> int:
    """Count nodes in a CSV file."""
    with open(filepath) as f:
        return sum(1 for _ in f) - 1  # -1 for header


def estimate_matrix_memory(num_nodes: int) -> float:
    """Estimate matrix memory in GB."""
    return (num_nodes ** 2 * 8) / (1024**3)


def run_experiment_for_size(size: str, nodes_file: str, edges_file: str, results: list):
    """Run complete experiment for a given network size."""
    
    print_header(f"EXPERIMENT: {size.upper()} NETWORK")
    
    # Count nodes to estimate memory
    num_nodes = count_nodes_in_file(nodes_file)
    matrix_gb = estimate_matrix_memory(num_nodes)
    
    print(f"Network has {num_nodes} nodes")
    print(f"Adjacency matrix would require: {matrix_gb:.2f} GB")
    
    matrix_feasible = matrix_gb < 20  # Conservative limit for 32GB machine
    
    if not matrix_feasible:
        print(f"\n⚠️  Matrix representation would use {matrix_gb:.1f} GB!")
        print("    This exceeds safe limits. Using adjacency list only.")
    
    result = {
        'size': size,
        'nodes': num_nodes,
        'matrix_memory_estimate_gb': matrix_gb,
        'list': {},
        'matrix': {}
    }
    
    # ========== ADJACENCY LIST ==========
    print_section("Adjacency List Representation")
    
    tracemalloc.start()
    start_time = time.time()
    
    list_graph = load_graph(nodes_file, edges_file, use_matrix=False)
    
    list_load_time = time.time() - start_time
    _, list_peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    result['list']['load_time_s'] = list_load_time
    result['list']['memory_mb'] = list_peak_memory / (1024*1024)
    result['edges'] = list_graph.num_edges
    
    print(f"Load time: {list_load_time:.3f} seconds")
    print(f"Peak memory: {list_peak_memory / (1024*1024):.2f} MB")
    
    # Benchmarks
    print("\nRunning edge queries benchmark (10,000 queries)...")
    eq_results = benchmark_edge_queries(list_graph, 10000)
    result['list']['edge_queries_per_sec'] = eq_results['queries_per_second']
    print(f"  Speed: {eq_results['queries_per_second']:.0f} queries/second")
    
    print("Running neighbor iteration benchmark (10,000 nodes)...")
    ni_results = benchmark_neighbor_iteration(list_graph, 10000)
    result['list']['neighbor_iter_time_s'] = ni_results['time_seconds']
    print(f"  Time: {ni_results['time_seconds']:.4f} seconds")
    
    print("Running shortest path benchmarks (5 random pairs)...")
    dj_results = benchmark_shortest_path(list_graph, 5, 'dijkstra')
    as_results = benchmark_shortest_path(list_graph, 5, 'astar')
    result['list']['dijkstra_avg_s'] = dj_results['avg_time_per_query']
    result['list']['astar_avg_s'] = as_results['avg_time_per_query']
    print(f"  Dijkstra avg: {dj_results['avg_time_per_query']:.4f}s ({dj_results['avg_nodes_explored']:.0f} nodes explored)")
    print(f"  A* avg: {as_results['avg_time_per_query']:.4f}s ({as_results['avg_nodes_explored']:.0f} nodes explored)")
    
    del list_graph
    
    # ========== ADJACENCY MATRIX ==========
    if matrix_feasible:
        print_section("Adjacency Matrix Representation")
        
        tracemalloc.start()
        start_time = time.time()
        
        try:
            matrix_graph = load_graph(nodes_file, edges_file, use_matrix=True)
            
            matrix_load_time = time.time() - start_time
            _, matrix_peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            result['matrix']['load_time_s'] = matrix_load_time
            result['matrix']['memory_mb'] = matrix_peak_memory / (1024*1024)
            
            print(f"Load time: {matrix_load_time:.3f} seconds")
            print(f"Peak memory: {matrix_peak_memory / (1024*1024):.2f} MB")
            
            # Benchmarks
            print("\nRunning edge queries benchmark (10,000 queries)...")
            eq_results = benchmark_edge_queries(matrix_graph, 10000)
            result['matrix']['edge_queries_per_sec'] = eq_results['queries_per_second']
            print(f"  Speed: {eq_results['queries_per_second']:.0f} queries/second")
            
            print("Running neighbor iteration benchmark (10,000 nodes)...")
            ni_results = benchmark_neighbor_iteration(matrix_graph, 10000)
            result['matrix']['neighbor_iter_time_s'] = ni_results['time_seconds']
            print(f"  Time: {ni_results['time_seconds']:.4f} seconds")
            
            print("Running shortest path benchmarks (5 random pairs)...")
            dj_results = benchmark_shortest_path(matrix_graph, 5, 'dijkstra')
            as_results = benchmark_shortest_path(matrix_graph, 5, 'astar')
            result['matrix']['dijkstra_avg_s'] = dj_results['avg_time_per_query']
            result['matrix']['astar_avg_s'] = as_results['avg_time_per_query']
            print(f"  Dijkstra avg: {dj_results['avg_time_per_query']:.4f}s ({dj_results['avg_nodes_explored']:.0f} nodes explored)")
            print(f"  A* avg: {as_results['avg_time_per_query']:.4f}s ({as_results['avg_nodes_explored']:.0f} nodes explored)")
            
            del matrix_graph
            
        except MemoryError as e:
            tracemalloc.stop()
            result['matrix']['error'] = str(e)
            print(f"\n❌ MEMORY ERROR: {e}")
            print("   This demonstrates why matrix representation fails at scale!")
    else:
        result['matrix']['skipped'] = True
        result['matrix']['reason'] = f"Would require {matrix_gb:.1f} GB RAM"
    
    results.append(result)
    return result


def print_comparison_table(results: list):
    """Print a comparison table of all results."""
    
    print_header("COMPARISON SUMMARY")
    
    # Header
    print(f"\n{'Size':<10} {'Nodes':<10} {'Edges':<10} {'List Mem':<12} {'Matrix Mem':<12} {'Ratio':<8}")
    print("-" * 70)
    
    for r in results:
        list_mem = r['list'].get('memory_mb', 0)
        matrix_mem = r['matrix'].get('memory_mb', 0) if not r['matrix'].get('skipped') else 0
        
        if matrix_mem > 0:
            ratio = f"{matrix_mem/list_mem:.1f}x"
        elif r['matrix'].get('skipped'):
            ratio = "N/A"
        else:
            ratio = "ERR"
        
        matrix_str = f"{matrix_mem:.1f} MB" if matrix_mem > 0 else (r['matrix'].get('reason', 'Error')[:10])
        
        print(f"{r['size']:<10} {r['nodes']:<10} {r['edges']:<10} {list_mem:.1f} MB      {matrix_str:<12} {ratio:<8}")
    
    # Performance comparison
    print(f"\n{'Size':<10} {'List Edge Q/s':<15} {'Matrix Edge Q/s':<15} {'Speedup':<10}")
    print("-" * 55)
    
    for r in results:
        list_eq = r['list'].get('edge_queries_per_sec', 0)
        matrix_eq = r['matrix'].get('edge_queries_per_sec', 0) if not r['matrix'].get('skipped') else 0
        
        if matrix_eq > 0:
            speedup = f"{matrix_eq/list_eq:.1f}x"
        else:
            speedup = "N/A"
        
        matrix_str = f"{matrix_eq:.0f}" if matrix_eq > 0 else "N/A"
        print(f"{r['size']:<10} {list_eq:.0f}           {matrix_str:<15} {speedup:<10}")
    
    # Dijkstra comparison
    print(f"\n{'Size':<10} {'List Dijkstra':<15} {'Matrix Dijkstra':<15} {'Slowdown':<10}")
    print("-" * 55)
    
    for r in results:
        list_dj = r['list'].get('dijkstra_avg_s', 0)
        matrix_dj = r['matrix'].get('dijkstra_avg_s', 0) if not r['matrix'].get('skipped') else 0
        
        if matrix_dj > 0 and list_dj > 0:
            slowdown = f"{matrix_dj/list_dj:.1f}x"
        else:
            slowdown = "N/A"
        
        matrix_str = f"{matrix_dj:.4f}s" if matrix_dj > 0 else "N/A"
        print(f"{r['size']:<10} {list_dj:.4f}s        {matrix_str:<15} {slowdown:<10}")


def save_results_csv(results: list, filename: str = "experiment_results.csv"):
    """Save results to CSV for later analysis."""
    filepath = get_data_path(filename)
    
    with open(filepath, 'w', newline='') as f:
        fieldnames = [
            'size', 'nodes', 'edges',
            'list_load_time_s', 'list_memory_mb', 'list_edge_qps', 
            'list_neighbor_time_s', 'list_dijkstra_s', 'list_astar_s',
            'matrix_load_time_s', 'matrix_memory_mb', 'matrix_edge_qps',
            'matrix_neighbor_time_s', 'matrix_dijkstra_s', 'matrix_astar_s',
            'matrix_skipped', 'matrix_reason'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in results:
            row = {
                'size': r['size'],
                'nodes': r['nodes'],
                'edges': r.get('edges', 0),
                'list_load_time_s': r['list'].get('load_time_s', ''),
                'list_memory_mb': r['list'].get('memory_mb', ''),
                'list_edge_qps': r['list'].get('edge_queries_per_sec', ''),
                'list_neighbor_time_s': r['list'].get('neighbor_iter_time_s', ''),
                'list_dijkstra_s': r['list'].get('dijkstra_avg_s', ''),
                'list_astar_s': r['list'].get('astar_avg_s', ''),
                'matrix_load_time_s': r['matrix'].get('load_time_s', ''),
                'matrix_memory_mb': r['matrix'].get('memory_mb', ''),
                'matrix_edge_qps': r['matrix'].get('edge_queries_per_sec', ''),
                'matrix_neighbor_time_s': r['matrix'].get('neighbor_iter_time_s', ''),
                'matrix_dijkstra_s': r['matrix'].get('dijkstra_avg_s', ''),
                'matrix_astar_s': r['matrix'].get('astar_avg_s', ''),
                'matrix_skipped': r['matrix'].get('skipped', False),
                'matrix_reason': r['matrix'].get('reason', '')
            }
            writer.writerow(row)
    
    print(f"\nResults saved to: {filepath}")


def main():
    print_header("CS3050 LAB 6: GRAPH REPRESENTATION EXPERIMENTS")
    
    print("""
This experiment will help you understand the tradeoffs between
adjacency matrix and adjacency list representations.

You will observe:
  • Memory usage scaling (O(V+E) vs O(V²))
  • Edge query performance (O(degree) vs O(1))
  • Neighbor iteration (O(degree) vs O(V))
  • How these affect real algorithm performance

Let's begin!
    """)
    
    # Check what data files are available
    sizes = ['tiny', 'small', 'medium', 'large', 'huge']
    available = []
    
    print("Checking for available network data...")
    for size in sizes:
        nodes_file, edges_file, exists = check_data_files(size)
        if exists:
            num_nodes = count_nodes_in_file(nodes_file)
            available.append((size, nodes_file, edges_file, num_nodes))
            print(f"  ✓ {size}: {num_nodes} nodes")
        else:
            print(f"  ✗ {size}: not generated yet")
    
    if not available:
        print("\n⚠️  No network data found!")
        print("Run the data generator first:")
        print("  cd ../scripts")
        print("  python generate_network.py --size tiny")
        print("  python generate_network.py --size small")
        print("  python generate_network.py --size medium")
        print("  # etc...")
        return
    
    print(f"\nFound {len(available)} network(s) to test.")
    
    # Run experiments
    results = []
    
    for size, nodes_file, edges_file, num_nodes in available:
        input(f"\nPress Enter to run experiment on {size.upper()} ({num_nodes} nodes)...")
        run_experiment_for_size(size, nodes_file, edges_file, results)
    
    # Print comparison
    if len(results) > 1:
        print_comparison_table(results)
    
    # Save results
    save_results_csv(results)
    
    # Discussion questions
    print_header("DISCUSSION QUESTIONS")
    print("""
Based on your experimental results, answer the following:

1. MEMORY SCALING
   - How does memory usage scale as network size increases?
   - At what point does the matrix become impractical?
   - Calculate: For a network with 200,000 nodes, how much RAM
     would the matrix require?

2. EDGE QUERY PERFORMANCE  
   - Which representation is faster for checking edge existence?
   - Why is this? Explain in terms of data structure access patterns.
   - When would this matter in a real application?

3. NEIGHBOR ITERATION
   - Which representation is faster for iterating over neighbors?
   - How does this affect Dijkstra's algorithm performance?
   - Why does the matrix get slower as V increases?

4. ALGORITHM IMPLICATIONS
   - Dijkstra's calls get_neighbors() many times. Which representation
     works better?
   - If you had an algorithm that only needed to check edge existence
     (not iterate neighbors), which would you choose?

5. REAL-WORLD TRADEOFFS
   - Road networks have ~4-6 edges per node on average. Is this "sparse"?
   - Social networks might have millions of nodes. Which representation
     would you use?
   - A clique of 1000 nodes has 500,000 edges. Which representation?

6. THE CROSSOVER POINT
   - At what graph density does matrix become worthwhile?
   - For your experimental data, plot memory_ratio vs edge_density.

Submit your answers with supporting data from your experiments.
    """)


if __name__ == '__main__':
    main()
