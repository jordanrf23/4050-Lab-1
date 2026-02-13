#!/usr/bin/env python3
"""
Graph Representations Module for CS3050 Lab 6

This module provides two graph representations for comparison:
  1. AdjacencyListGraph - O(V + E) memory, fast neighbor iteration
  2. AdjacencyMatrixGraph - O(V²) memory, O(1) edge lookup

Students will implement algorithms using both representations and
measure the performance differences on various network sizes.
"""

import csv
import heapq
import math
import time
import sys
import tracemalloc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict


@dataclass
class Node:
    """Represents a node with geographic coordinates."""
    id: int
    lat: float
    lon: float


@dataclass
class Edge:
    """Represents a weighted edge."""
    to_node: int
    weight: float


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance in km between two points."""
    R = 6371  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))


# ============================================================================
# ABSTRACT BASE CLASS - Defines the interface both representations must implement
# ============================================================================

class Graph(ABC):
    """Abstract base class for graph representations."""
    
    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.num_edges = 0
    
    @abstractmethod
    def add_edge(self, from_id: int, to_id: int, weight: float) -> None:
        """Add a weighted edge to the graph."""
        pass
    
    @abstractmethod
    def get_neighbors(self, node_id: int) -> List[Tuple[int, float]]:
        """Return list of (neighbor_id, weight) tuples."""
        pass
    
    @abstractmethod
    def has_edge(self, from_id: int, to_id: int) -> bool:
        """Check if an edge exists between two nodes."""
        pass
    
    @abstractmethod
    def get_edge_weight(self, from_id: int, to_id: int) -> Optional[float]:
        """Get weight of edge, or None if no edge exists."""
        pass
    
    @abstractmethod
    def get_memory_bytes(self) -> int:
        """Estimate memory usage of the graph structure."""
        pass
    
    def add_node(self, node_id: int, lat: float, lon: float) -> None:
        """Add a node with coordinates."""
        self.nodes[node_id] = Node(node_id, lat, lon)
    
    def get_node(self, node_id: int) -> Optional[Node]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def num_nodes(self) -> int:
        """Return number of nodes."""
        return len(self.nodes)


# ============================================================================
# ADJACENCY LIST IMPLEMENTATION
# ============================================================================

class AdjacencyListGraph(Graph):
    """
    Graph using adjacency list representation.
    
    Memory: O(V + E)
    Add edge: O(1)
    Get neighbors: O(degree) to iterate
    Has edge: O(degree) to check
    
    Best for: Sparse graphs, algorithms that iterate over neighbors
    """
    
    def __init__(self):
        super().__init__()
        # Dict mapping node_id -> list of (neighbor_id, weight)
        self.adj_list: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    
    def add_edge(self, from_id: int, to_id: int, weight: float) -> None:
        """Add undirected edge."""
        self.adj_list[from_id].append((to_id, weight))
        self.adj_list[to_id].append((from_id, weight))
        self.num_edges += 1
    
    def get_neighbors(self, node_id: int) -> List[Tuple[int, float]]:
        """Return list of (neighbor_id, weight) tuples."""
        return self.adj_list[node_id]
    
    def has_edge(self, from_id: int, to_id: int) -> bool:
        """Check if edge exists. O(degree) operation."""
        for neighbor, _ in self.adj_list[from_id]:
            if neighbor == to_id:
                return True
        return False
    
    def get_edge_weight(self, from_id: int, to_id: int) -> Optional[float]:
        """Get edge weight. O(degree) operation."""
        for neighbor, weight in self.adj_list[from_id]:
            if neighbor == to_id:
                return weight
        return None
    
    def get_memory_bytes(self) -> int:
        """Estimate memory usage."""
        # Base dict overhead + per-node list overhead + edge entries
        # Each edge stored twice (undirected)
        node_overhead = len(self.nodes) * 64  # dict entry + list object
        edge_storage = self.num_edges * 2 * 16  # (int, float) tuple each direction
        node_data = len(self.nodes) * 32  # Node dataclass
        return node_overhead + edge_storage + node_data


# ============================================================================
# ADJACENCY MATRIX IMPLEMENTATION
# ============================================================================

class AdjacencyMatrixGraph(Graph):
    """
    Graph using adjacency matrix representation.
    
    Memory: O(V²)
    Add edge: O(1)
    Get neighbors: O(V) to find all non-zero entries
    Has edge: O(1)
    
    Best for: Dense graphs, frequent edge existence queries
    
    WARNING: For V > ~50,000 nodes, this will use >20GB of RAM!
    """
    
    def __init__(self, max_nodes: int = 0):
        super().__init__()
        self.max_nodes = max_nodes
        # 2D list representing the adjacency matrix
        # matrix[i][j] = weight if edge exists, 0 otherwise
        # We'll initialize lazily or require max_nodes upfront
        self.matrix: Optional[List[List[float]]] = None
        self._initialized = False
    
    def _ensure_matrix(self, size: int) -> None:
        """Initialize matrix if needed."""
        if not self._initialized or (self.matrix and len(self.matrix) < size):
            print(f"  Allocating {size}x{size} matrix ({size*size*8/(1024**3):.2f} GB)...")
            self.max_nodes = size
            # Allocate the matrix - this is where memory usage explodes!
            self.matrix = [[0.0] * size for _ in range(size)]
            self._initialized = True
            print("  Matrix allocated.")
    
    def add_node(self, node_id: int, lat: float, lon: float) -> None:
        """Add a node, expanding matrix if needed."""
        super().add_node(node_id, lat, lon)
        if node_id >= self.max_nodes:
            # For efficiency, we should know max_nodes upfront
            # This is a limitation of the matrix approach!
            pass
    
    def initialize_matrix(self, size: int) -> None:
        """Pre-allocate matrix of given size. Call this before adding edges."""
        self._ensure_matrix(size)
    
    def add_edge(self, from_id: int, to_id: int, weight: float) -> None:
        """Add undirected edge."""
        if self.matrix is None:
            raise RuntimeError("Matrix not initialized! Call initialize_matrix() first.")
        self.matrix[from_id][to_id] = weight
        self.matrix[to_id][from_id] = weight
        self.num_edges += 1
    
    def get_neighbors(self, node_id: int) -> List[Tuple[int, float]]:
        """Return list of neighbors. O(V) operation - must scan entire row!"""
        if self.matrix is None:
            return []
        neighbors = []
        for i, weight in enumerate(self.matrix[node_id]):
            if weight > 0:
                neighbors.append((i, weight))
        return neighbors
    
    def has_edge(self, from_id: int, to_id: int) -> bool:
        """Check if edge exists. O(1) operation - the matrix advantage!"""
        if self.matrix is None:
            return False
        return self.matrix[from_id][to_id] > 0
    
    def get_edge_weight(self, from_id: int, to_id: int) -> Optional[float]:
        """Get edge weight. O(1) operation."""
        if self.matrix is None:
            return None
        weight = self.matrix[from_id][to_id]
        return weight if weight > 0 else None
    
    def get_memory_bytes(self) -> int:
        """Estimate memory usage."""
        if self.matrix is None:
            return len(self.nodes) * 32
        # Matrix: n*n floats (8 bytes each in Python lists)
        # Plus list overhead
        n = len(self.matrix)
        matrix_bytes = n * n * 8 + n * 56  # 56 bytes per list object overhead
        node_data = len(self.nodes) * 32
        return matrix_bytes + node_data


# ============================================================================
# GRAPH LOADING FUNCTIONS
# ============================================================================

def load_nodes(filepath: str, graph: Graph) -> int:
    """Load nodes from CSV file."""
    count = 0
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = int(row['id'])
            lat = float(row['lat'])
            lon = float(row['lon'])
            graph.add_node(node_id, lat, lon)
            count += 1
    return count


def load_edges(filepath: str, graph: Graph) -> int:
    """Load edges from CSV file."""
    count = 0
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            from_id = int(row['from'])
            to_id = int(row['to'])
            weight = float(row['distance'])
            graph.add_edge(from_id, to_id, weight)
            count += 1
    return count


def load_graph(nodes_file: str, edges_file: str, use_matrix: bool = False) -> Graph:
    """
    Load a graph from CSV files.
    
    Args:
        nodes_file: Path to nodes CSV
        edges_file: Path to edges CSV
        use_matrix: If True, use adjacency matrix; otherwise adjacency list
    
    Returns:
        Loaded Graph object
    """
    print(f"\nLoading graph ({'matrix' if use_matrix else 'list'} representation)...")
    
    # First, count nodes to pre-allocate matrix
    if use_matrix:
        node_count = sum(1 for _ in open(nodes_file)) - 1  # -1 for header
        print(f"  Detected {node_count} nodes")
        
        # Check if this will fit in memory
        matrix_gb = (node_count ** 2 * 8) / (1024**3)
        if matrix_gb > 25:
            raise MemoryError(
                f"Matrix would require ~{matrix_gb:.1f}GB RAM! "
                f"Use adjacency list for graphs with >{int(math.sqrt(25 * 1024**3 / 8))} nodes."
            )
        
        graph = AdjacencyMatrixGraph(node_count)
        graph.initialize_matrix(node_count)
    else:
        graph = AdjacencyListGraph()
    
    start_time = time.time()
    
    num_nodes = load_nodes(nodes_file, graph)
    load_node_time = time.time() - start_time
    print(f"  Loaded {num_nodes} nodes in {load_node_time:.3f}s")
    
    edge_start = time.time()
    num_edges = load_edges(edges_file, graph)
    load_edge_time = time.time() - edge_start
    print(f"  Loaded {num_edges} edges in {load_edge_time:.3f}s")
    
    total_time = time.time() - start_time
    memory_mb = graph.get_memory_bytes() / (1024 * 1024)
    
    print(f"  Total load time: {total_time:.3f}s")
    print(f"  Estimated memory: {memory_mb:.2f} MB")
    
    return graph


# ============================================================================
# SHORTEST PATH ALGORITHMS
# ============================================================================

@dataclass(order=True)
class PQEntry:
    """Priority queue entry for Dijkstra/A*."""
    priority: float
    node_id: int = field(compare=False)


def dijkstra(graph: Graph, start: int, goal: int) -> Tuple[List[int], float, int]:
    """
    Dijkstra's algorithm for shortest path.
    
    Returns:
        (path, total_distance, nodes_explored)
    """
    if start not in graph.nodes or goal not in graph.nodes:
        return [], float('inf'), 0
    
    distances: Dict[int, float] = {start: 0}
    previous: Dict[int, int] = {}
    visited: Set[int] = set()
    pq = [PQEntry(0, start)]
    nodes_explored = 0
    
    while pq:
        entry = heapq.heappop(pq)
        current = entry.node_id
        
        if current in visited:
            continue
        
        visited.add(current)
        nodes_explored += 1
        
        if current == goal:
            break
        
        # This is where representation matters!
        # Matrix: O(V) to get neighbors
        # List: O(degree) to get neighbors
        for neighbor, weight in graph.get_neighbors(current):
            if neighbor in visited:
                continue
            
            new_dist = distances[current] + weight
            
            if neighbor not in distances or new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = current
                heapq.heappush(pq, PQEntry(new_dist, neighbor))
    
    # Reconstruct path
    if goal not in previous and goal != start:
        return [], float('inf'), nodes_explored
    
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = previous.get(current)
    path.reverse()
    
    return path, distances.get(goal, float('inf')), nodes_explored


def astar(graph: Graph, start: int, goal: int) -> Tuple[List[int], float, int]:
    """
    A* algorithm using haversine heuristic.
    
    Returns:
        (path, total_distance, nodes_explored)
    """
    if start not in graph.nodes or goal not in graph.nodes:
        return [], float('inf'), 0
    
    goal_node = graph.get_node(goal)
    if goal_node is None:
        return [], float('inf'), 0
    
    def heuristic(node_id: int) -> float:
        node = graph.get_node(node_id)
        if node is None:
            return float('inf')
        return haversine_distance(node.lat, node.lon, goal_node.lat, goal_node.lon)
    
    g_scores: Dict[int, float] = {start: 0}
    f_scores: Dict[int, float] = {start: heuristic(start)}
    previous: Dict[int, int] = {}
    visited: Set[int] = set()
    pq = [PQEntry(f_scores[start], start)]
    nodes_explored = 0
    
    while pq:
        entry = heapq.heappop(pq)
        current = entry.node_id
        
        if current in visited:
            continue
        
        visited.add(current)
        nodes_explored += 1
        
        if current == goal:
            break
        
        for neighbor, weight in graph.get_neighbors(current):
            if neighbor in visited:
                continue
            
            tentative_g = g_scores[current] + weight
            
            if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g
                f_scores[neighbor] = tentative_g + heuristic(neighbor)
                previous[neighbor] = current
                heapq.heappush(pq, PQEntry(f_scores[neighbor], neighbor))
    
    # Reconstruct path
    if goal not in previous and goal != start:
        return [], float('inf'), nodes_explored
    
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = previous.get(current)
    path.reverse()
    
    return path, g_scores.get(goal, float('inf')), nodes_explored


# ============================================================================
# BENCHMARKING FUNCTIONS
# ============================================================================

def benchmark_edge_queries(graph: Graph, num_queries: int = 10000) -> Dict[str, float]:
    """Benchmark random edge existence queries."""
    import random
    
    node_ids = list(graph.nodes.keys())
    if len(node_ids) < 2:
        return {'queries': 0, 'time': 0, 'qps': 0}
    
    # Generate random pairs
    pairs = [(random.choice(node_ids), random.choice(node_ids)) for _ in range(num_queries)]
    
    start = time.time()
    hits = 0
    for a, b in pairs:
        if graph.has_edge(a, b):
            hits += 1
    elapsed = time.time() - start
    
    return {
        'queries': num_queries,
        'time_seconds': elapsed,
        'queries_per_second': num_queries / elapsed if elapsed > 0 else 0,
        'edge_density': hits / num_queries
    }


def benchmark_neighbor_iteration(graph: Graph, num_iterations: int = 1000) -> Dict[str, float]:
    """Benchmark neighbor iteration."""
    import random
    
    node_ids = list(graph.nodes.keys())
    if not node_ids:
        return {'iterations': 0, 'time': 0}
    
    nodes_to_visit = [random.choice(node_ids) for _ in range(num_iterations)]
    
    start = time.time()
    total_neighbors = 0
    for node_id in nodes_to_visit:
        neighbors = graph.get_neighbors(node_id)
        total_neighbors += len(neighbors)
    elapsed = time.time() - start
    
    return {
        'iterations': num_iterations,
        'time_seconds': elapsed,
        'total_neighbors_visited': total_neighbors,
        'avg_degree': total_neighbors / num_iterations if num_iterations > 0 else 0
    }


def benchmark_shortest_path(graph: Graph, num_queries: int = 10, algorithm: str = 'dijkstra') -> Dict[str, float]:
    """Benchmark shortest path queries."""
    import random
    
    node_ids = list(graph.nodes.keys())
    if len(node_ids) < 2:
        return {'queries': 0, 'time': 0}
    
    # Generate random source-destination pairs
    # Spread them across the graph for meaningful paths
    pairs = []
    for _ in range(num_queries):
        start = random.choice(node_ids)
        goal = random.choice(node_ids)
        while goal == start:
            goal = random.choice(node_ids)
        pairs.append((start, goal))
    
    func = dijkstra if algorithm == 'dijkstra' else astar
    
    start_time = time.time()
    total_nodes_explored = 0
    paths_found = 0
    
    for start, goal in pairs:
        path, dist, explored = func(graph, start, goal)
        total_nodes_explored += explored
        if path:
            paths_found += 1
    
    elapsed = time.time() - start_time
    
    return {
        'queries': num_queries,
        'algorithm': algorithm,
        'time_seconds': elapsed,
        'avg_time_per_query': elapsed / num_queries if num_queries > 0 else 0,
        'paths_found': paths_found,
        'total_nodes_explored': total_nodes_explored,
        'avg_nodes_explored': total_nodes_explored / num_queries if num_queries > 0 else 0
    }


def run_full_benchmark(nodes_file: str, edges_file: str) -> Dict:
    """Run comprehensive benchmarks comparing both representations."""
    results = {
        'adjacency_list': {},
        'adjacency_matrix': {}
    }
    
    # First count nodes to check if matrix is feasible
    with open(nodes_file) as f:
        node_count = sum(1 for _ in f) - 1
    
    matrix_gb = (node_count ** 2 * 8) / (1024**3)
    matrix_feasible = matrix_gb < 25
    
    print(f"\n{'='*60}")
    print(f"BENCHMARKING GRAPH REPRESENTATIONS")
    print(f"Nodes: {node_count}, Matrix would use: {matrix_gb:.2f} GB")
    print(f"Matrix feasible: {matrix_feasible}")
    print(f"{'='*60}")
    
    # Test Adjacency List
    print("\n--- Adjacency List ---")
    tracemalloc.start()
    
    list_load_start = time.time()
    list_graph = load_graph(nodes_file, edges_file, use_matrix=False)
    list_load_time = time.time() - list_load_start
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    results['adjacency_list']['load_time'] = list_load_time
    results['adjacency_list']['memory_mb'] = peak / (1024 * 1024)
    results['adjacency_list']['estimated_memory_mb'] = list_graph.get_memory_bytes() / (1024 * 1024)
    
    print(f"Load time: {list_load_time:.3f}s")
    print(f"Peak memory: {peak / (1024*1024):.2f} MB")
    
    print("\nBenchmarking edge queries...")
    results['adjacency_list']['edge_queries'] = benchmark_edge_queries(list_graph, 10000)
    print(f"  {results['adjacency_list']['edge_queries']['queries_per_second']:.0f} queries/sec")
    
    print("Benchmarking neighbor iteration...")
    results['adjacency_list']['neighbor_iteration'] = benchmark_neighbor_iteration(list_graph, 10000)
    print(f"  {results['adjacency_list']['neighbor_iteration']['time_seconds']:.3f}s for 10k iterations")
    
    print("Benchmarking Dijkstra...")
    results['adjacency_list']['dijkstra'] = benchmark_shortest_path(list_graph, 5, 'dijkstra')
    print(f"  Avg time: {results['adjacency_list']['dijkstra']['avg_time_per_query']:.3f}s")
    
    print("Benchmarking A*...")
    results['adjacency_list']['astar'] = benchmark_shortest_path(list_graph, 5, 'astar')
    print(f"  Avg time: {results['adjacency_list']['astar']['avg_time_per_query']:.3f}s")
    
    del list_graph
    
    # Test Adjacency Matrix (if feasible)
    if matrix_feasible:
        print("\n--- Adjacency Matrix ---")
        tracemalloc.start()
        
        matrix_load_start = time.time()
        matrix_graph = load_graph(nodes_file, edges_file, use_matrix=True)
        matrix_load_time = time.time() - matrix_load_start
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        results['adjacency_matrix']['load_time'] = matrix_load_time
        results['adjacency_matrix']['memory_mb'] = peak / (1024 * 1024)
        results['adjacency_matrix']['estimated_memory_mb'] = matrix_graph.get_memory_bytes() / (1024 * 1024)
        
        print(f"Load time: {matrix_load_time:.3f}s")
        print(f"Peak memory: {peak / (1024*1024):.2f} MB")
        
        print("\nBenchmarking edge queries...")
        results['adjacency_matrix']['edge_queries'] = benchmark_edge_queries(matrix_graph, 10000)
        print(f"  {results['adjacency_matrix']['edge_queries']['queries_per_second']:.0f} queries/sec")
        
        print("Benchmarking neighbor iteration...")
        results['adjacency_matrix']['neighbor_iteration'] = benchmark_neighbor_iteration(matrix_graph, 10000)
        print(f"  {results['adjacency_matrix']['neighbor_iteration']['time_seconds']:.3f}s for 10k iterations")
        
        print("Benchmarking Dijkstra...")
        results['adjacency_matrix']['dijkstra'] = benchmark_shortest_path(matrix_graph, 5, 'dijkstra')
        print(f"  Avg time: {results['adjacency_matrix']['dijkstra']['avg_time_per_query']:.3f}s")
        
        print("Benchmarking A*...")
        results['adjacency_matrix']['astar'] = benchmark_shortest_path(matrix_graph, 5, 'astar')
        print(f"  Avg time: {results['adjacency_matrix']['astar']['avg_time_per_query']:.3f}s")
        
        del matrix_graph
    else:
        print("\n--- Adjacency Matrix ---")
        print(f"SKIPPED: Would require {matrix_gb:.1f} GB RAM")
        results['adjacency_matrix']['skipped'] = True
        results['adjacency_matrix']['reason'] = f"Would require {matrix_gb:.1f} GB RAM"
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    list_mem = results['adjacency_list']['memory_mb']
    print(f"\nAdjacency List:")
    print(f"  Memory: {list_mem:.2f} MB")
    print(f"  Load time: {results['adjacency_list']['load_time']:.3f}s")
    print(f"  Edge query speed: {results['adjacency_list']['edge_queries']['queries_per_second']:.0f}/s")
    
    if not results['adjacency_matrix'].get('skipped'):
        matrix_mem = results['adjacency_matrix']['memory_mb']
        print(f"\nAdjacency Matrix:")
        print(f"  Memory: {matrix_mem:.2f} MB ({matrix_mem/list_mem:.1f}x more)")
        print(f"  Load time: {results['adjacency_matrix']['load_time']:.3f}s")
        print(f"  Edge query speed: {results['adjacency_matrix']['edge_queries']['queries_per_second']:.0f}/s")
        
        # Compare
        list_edge_qps = results['adjacency_list']['edge_queries']['queries_per_second']
        matrix_edge_qps = results['adjacency_matrix']['edge_queries']['queries_per_second']
        print(f"\nEdge query speedup (matrix/list): {matrix_edge_qps/list_edge_qps:.2f}x")
        
        list_neighbor_time = results['adjacency_list']['neighbor_iteration']['time_seconds']
        matrix_neighbor_time = results['adjacency_matrix']['neighbor_iteration']['time_seconds']
        print(f"Neighbor iteration slowdown (matrix/list): {matrix_neighbor_time/list_neighbor_time:.2f}x")
    
    return results


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python graph_representations.py <nodes.csv> <edges.csv> [--benchmark]")
        print("       python graph_representations.py <nodes.csv> <edges.csv> <start> <goal>")
        sys.exit(1)
    
    nodes_file = sys.argv[1]
    edges_file = sys.argv[2]
    
    if len(sys.argv) > 3 and sys.argv[3] == '--benchmark':
        results = run_full_benchmark(nodes_file, edges_file)
    elif len(sys.argv) >= 5:
        start_node = int(sys.argv[3])
        goal_node = int(sys.argv[4])
        
        graph = load_graph(nodes_file, edges_file, use_matrix=False)
        
        print(f"\n=== Dijkstra: {start_node} -> {goal_node} ===")
        path, dist, explored = dijkstra(graph, start_node, goal_node)
        if path:
            print(f"Path: {' -> '.join(map(str, path[:10]))}{'...' if len(path) > 10 else ''}")
            print(f"Distance: {dist:.2f} km")
            print(f"Nodes explored: {explored}")
        else:
            print("No path found!")
        
        print(f"\n=== A*: {start_node} -> {goal_node} ===")
        path, dist, explored = astar(graph, start_node, goal_node)
        if path:
            print(f"Path: {' -> '.join(map(str, path[:10]))}{'...' if len(path) > 10 else ''}")
            print(f"Distance: {dist:.2f} km")
            print(f"Nodes explored: {explored}")
        else:
            print("No path found!")
    else:
        print("Loading graph with adjacency list...")
        graph = load_graph(nodes_file, edges_file, use_matrix=False)
        print(f"\nGraph loaded: {graph.num_nodes()} nodes, {graph.num_edges} edges")
