#!/usr/bin/env python3
"""
Network Data Generator for CS3050 Lab 6
Generates synthetic road networks at various scales to demonstrate
adjacency matrix vs adjacency list performance characteristics.

Usage:
    python generate_network.py --size tiny      # ~100 nodes (baseline)
    python generate_network.py --size small     # ~1,000 nodes
    python generate_network.py --size medium    # ~10,000 nodes
    python generate_network.py --size large     # ~50,000 nodes
    python generate_network.py --size huge      # ~100,000 nodes (list only!)
    python generate_network.py --nodes 5000 --avg-degree 6  # custom
"""

import argparse
import csv
import random
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Set

# Predefined network sizes
NETWORK_SIZES = {
    'tiny':   {'nodes': 100,    'avg_degree': 4,  'description': 'Baseline - no visible difference'},
    'small':  {'nodes': 1000,   'avg_degree': 5,  'description': 'Small town road network'},
    'medium': {'nodes': 10000,  'avg_degree': 6,  'description': 'City road network - matrix starts to show strain'},
    'large':  {'nodes': 50000,  'avg_degree': 6,  'description': 'Metro area - matrix uses ~10GB RAM'},
    'huge':   {'nodes': 100000, 'avg_degree': 6,  'description': 'Regional network - matrix WILL crash 32GB machine!'},
}


@dataclass
class Node:
    id: int
    lat: float
    lon: float


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in km between two lat/lon points."""
    R = 6371  # Earth's radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def generate_grid_network(
    num_nodes: int,
    avg_degree: float,
    center_lat: float = 38.9,
    center_lon: float = -77.0,
    spread_km: float = 50.0
) -> Tuple[List[Node], List[Tuple[int, int, float]]]:
    """
    Generate a realistic road-like network using a grid with perturbation.
    
    This creates a network that resembles real road networks:
    - Nodes are roughly grid-aligned (like city blocks)
    - Edges connect nearby nodes (like roads)
    - Some randomness for realism
    """
    # Calculate grid dimensions
    grid_side = int(math.sqrt(num_nodes))
    actual_nodes = grid_side * grid_side
    
    # Convert spread from km to degrees (approximate)
    spread_deg = spread_km / 111.0  # ~111 km per degree
    
    nodes: List[Node] = []
    node_positions: List[Tuple[float, float]] = []
    
    print(f"Generating {actual_nodes} nodes in a {grid_side}x{grid_side} grid...")
    
    # Generate nodes on a perturbed grid
    for i in range(grid_side):
        for j in range(grid_side):
            node_id = i * grid_side + j
            
            # Base grid position
            base_lat = center_lat + (i / grid_side - 0.5) * spread_deg
            base_lon = center_lon + (j / grid_side - 0.5) * spread_deg
            
            # Add perturbation (up to 20% of cell size)
            perturbation = spread_deg / grid_side * 0.2
            lat = base_lat + random.uniform(-perturbation, perturbation)
            lon = base_lon + random.uniform(-perturbation, perturbation)
            
            nodes.append(Node(node_id, lat, lon))
            node_positions.append((lat, lon))
    
    # Generate edges
    edges: List[Tuple[int, int, float]] = []
    edge_set: Set[Tuple[int, int]] = set()
    
    # Target number of edges based on average degree
    target_edges = int(actual_nodes * avg_degree / 2)
    
    print(f"Generating ~{target_edges} edges (avg degree {avg_degree})...")
    
    # First, connect grid neighbors (creates the road network backbone)
    for i in range(grid_side):
        for j in range(grid_side):
            node_id = i * grid_side + j
            
            # Connect to right neighbor
            if j < grid_side - 1:
                neighbor_id = i * grid_side + (j + 1)
                if (node_id, neighbor_id) not in edge_set and (neighbor_id, node_id) not in edge_set:
                    dist = haversine_distance(
                        nodes[node_id].lat, nodes[node_id].lon,
                        nodes[neighbor_id].lat, nodes[neighbor_id].lon
                    )
                    edges.append((node_id, neighbor_id, dist))
                    edge_set.add((node_id, neighbor_id))
            
            # Connect to bottom neighbor
            if i < grid_side - 1:
                neighbor_id = (i + 1) * grid_side + j
                if (node_id, neighbor_id) not in edge_set and (neighbor_id, node_id) not in edge_set:
                    dist = haversine_distance(
                        nodes[node_id].lat, nodes[node_id].lon,
                        nodes[neighbor_id].lat, nodes[neighbor_id].lon
                    )
                    edges.append((node_id, neighbor_id, dist))
                    edge_set.add((node_id, neighbor_id))
    
    # Add some diagonal connections for variety (simulates diagonal roads)
    diagonal_count = target_edges - len(edges)
    if diagonal_count > 0:
        print(f"Adding {diagonal_count} additional connections...")
        attempts = 0
        max_attempts = diagonal_count * 10
        
        while len(edges) < target_edges and attempts < max_attempts:
            attempts += 1
            # Pick a random node
            node_id = random.randint(0, actual_nodes - 1)
            i, j = node_id // grid_side, node_id % grid_side
            
            # Try to connect to a nearby node (within 2-3 grid cells)
            di = random.randint(-3, 3)
            dj = random.randint(-3, 3)
            
            if di == 0 and dj == 0:
                continue
            
            ni, nj = i + di, j + dj
            if 0 <= ni < grid_side and 0 <= nj < grid_side:
                neighbor_id = ni * grid_side + nj
                
                if (node_id, neighbor_id) not in edge_set and (neighbor_id, node_id) not in edge_set:
                    dist = haversine_distance(
                        nodes[node_id].lat, nodes[node_id].lon,
                        nodes[neighbor_id].lat, nodes[neighbor_id].lon
                    )
                    edges.append((node_id, neighbor_id, dist))
                    edge_set.add((node_id, neighbor_id))
    
    return nodes, edges


def save_network(nodes: List[Node], edges: List[Tuple[int, int, float]], output_dir: str, name: str):
    """Save network to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    nodes_file = os.path.join(output_dir, f'{name}_nodes.csv')
    edges_file = os.path.join(output_dir, f'{name}_edges.csv')
    
    print(f"Saving {len(nodes)} nodes to {nodes_file}")
    with open(nodes_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'lat', 'lon'])
        for node in nodes:
            writer.writerow([node.id, f'{node.lat:.6f}', f'{node.lon:.6f}'])
    
    print(f"Saving {len(edges)} edges to {edges_file}")
    with open(edges_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['from', 'to', 'distance'])
        for from_id, to_id, dist in edges:
            writer.writerow([from_id, to_id, f'{dist:.4f}'])
    
    # Also save metadata
    meta_file = os.path.join(output_dir, f'{name}_meta.txt')
    with open(meta_file, 'w') as f:
        f.write(f"Network: {name}\n")
        f.write(f"Nodes: {len(nodes)}\n")
        f.write(f"Edges: {len(edges)}\n")
        f.write(f"Average degree: {2 * len(edges) / len(nodes):.2f}\n")
        f.write(f"\nMemory estimates:\n")
        
        # Memory calculations
        # Adjacency matrix: n*n * 8 bytes (for float64)
        matrix_bytes = len(nodes) * len(nodes) * 8
        # Adjacency list: n * (pointer + list overhead) + e * (node_id + weight)
        list_bytes = len(nodes) * 64 + len(edges) * 2 * 16
        
        f.write(f"  Adjacency Matrix: {matrix_bytes / (1024**3):.2f} GB\n")
        f.write(f"  Adjacency List: {list_bytes / (1024**2):.2f} MB\n")
        f.write(f"  Ratio (Matrix/List): {matrix_bytes / list_bytes:.1f}x\n")
    
    print(f"Metadata saved to {meta_file}")
    return nodes_file, edges_file


def main():
    parser = argparse.ArgumentParser(
        description='Generate road network data for graph algorithm experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Network size descriptions:
  tiny   (100 nodes)    - Baseline, no visible performance difference
  small  (1,000 nodes)  - Small town, minimal difference
  medium (10,000 nodes) - City network, matrix shows memory growth
  large  (50,000 nodes) - Metro area, matrix ~10GB RAM, significant slowdown
  huge   (100,000 nodes)- Regional network, matrix crashes 32GB machines!
        """
    )
    
    parser.add_argument('--size', choices=NETWORK_SIZES.keys(),
                        help='Predefined network size')
    parser.add_argument('--nodes', type=int, help='Custom number of nodes')
    parser.add_argument('--avg-degree', type=float, default=6.0,
                        help='Average degree (edges per node)')
    parser.add_argument('--output', default='../data',
                        help='Output directory (default: ../data)')
    parser.add_argument('--name', help='Output file prefix (default: based on size)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    if args.size:
        config = NETWORK_SIZES[args.size]
        num_nodes = config['nodes']
        avg_degree = config['avg_degree']
        name = args.name or args.size
        print(f"\n=== Generating {args.size.upper()} network ===")
        print(f"Description: {config['description']}")
    elif args.nodes:
        num_nodes = args.nodes
        avg_degree = args.avg_degree
        name = args.name or f'custom_{num_nodes}'
        print(f"\n=== Generating custom network ===")
    else:
        parser.print_help()
        print("\nError: Specify either --size or --nodes")
        return
    
    # Calculate and show memory estimates upfront
    matrix_gb = (num_nodes ** 2 * 8) / (1024**3)
    print(f"\nExpected nodes: {num_nodes}")
    print(f"Adjacency matrix memory: {matrix_gb:.2f} GB")
    
    if matrix_gb > 20:
        print("\n⚠️  WARNING: Matrix representation will use >20GB RAM!")
        print("    Consider using adjacency list only for this size.")
    
    nodes, edges = generate_grid_network(num_nodes, avg_degree)
    save_network(nodes, edges, args.output, name)
    
    print(f"\n✓ Network generation complete!")
    print(f"  Actual nodes: {len(nodes)}")
    print(f"  Actual edges: {len(edges)}")
    print(f"  Actual avg degree: {2 * len(edges) / len(nodes):.2f}")


if __name__ == '__main__':
    main()
