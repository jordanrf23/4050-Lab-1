## C Implementation

The C implementation in `c/graph_benchmark.c` contains both graph representations and Dijkstra's algorithm in a single file (~500 lines). Here's how it's organized:

**Data Structures (lines 30-70):**
- `Node` — stores id, latitude, longitude
- `EdgeNode` — linked list node for adjacency list edges
- `AdjListGraph` — array of linked lists
- `AdjMatrixGraph` — 2D array of weights

**Adjacency List Functions (lines 150-230):**
- `adjlist_create()`, `adjlist_add_edge()`, `adjlist_has_edge()`
- Neighbor iteration is implicit in `dijkstra_adjlist()` — it walks the linked list

**Adjacency Matrix Functions (lines 235-310):**
- `adjmatrix_create()` — this is where the V² memory allocation happens
- Neighbor iteration scans the entire row checking for non-zero weights

**Dijkstra's Algorithm (lines 400-500):**
- `dijkstra_adjlist()` — inner loop walks linked list (fast)
- `dijkstra_adjmatrix()` — inner loop scans all V entries (slow)

**Compiling and Running:**

```bash
cd c
make                    # compiles to ./graph_benchmark

# Run on small network
./graph_benchmark ../data/small_nodes.csv ../data/small_edges.csv

# Run on medium network (watch the matrix allocation time!)
./graph_benchmark ../data/medium_nodes.csv ../data/medium_edges.csv
```

The C version shows even more dramatic performance differences than Python because there's less interpreter overhead — the algorithmic complexity dominates.