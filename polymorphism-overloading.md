# Polymorphism, Overloading and What's Happening Here

## A few notes about how "get_neighbors" works in Python 

The get_neighbors() method is in python/graph_representations.py — both graph classes implement it:

Adjacency List (lines ~130-133):

```
pythondef get_neighbors(self, node_id: int) -> List[Tuple[int, float]]:
    """Return list of (neighbor_id, weight) tuples."""
    return self.adj_list[node_id]
Adjacency Matrix (lines ~175-182):
pythondef get_neighbors(self, node_id: int) -> List[Tuple[int, float]]:
    """Return list of neighbors. O(V) operation - must scan entire row!"""
    if self.matrix is None:
        return []
    neighbors = []
    for i, weight in enumerate(self.matrix[node_id]):
        if weight > 0:
            neighbors.append((i, weight))
    return neighbors
```

The contrast is the whole point of the lab — list returns the neighbor list directly (O(degree)), while matrix must scan all V entries checking for non-zero weights (O(V)).

In the C implementation (c/graph_benchmark.c), this logic is inlined directly in the Dijkstra functions rather than abstracted into a separate function — look for the comments 
```
/* Iterate neighbors - this is O(degree) for adjacency list */ around line 420 and 
/* Iterate neighbors - this is O(V) for adjacency matrix! */ around line 470.
```

In Python, this isn't overloading - it's polymorphism through inheritance. Both AdjacencyListGraph and AdjacencyMatrixGraph inherit from the abstract base class Graph, which defines get_neighbors() as an abstract method. Each subclass provides its own implementation.
When you call `graph.get_neighbors(node_id)`, Python looks up the method on the actual object's class, not the variable's declared type. So if graph is an AdjacencyListGraph instance, it calls `AdjacencyListGraph.get_neighbors()`. If it's an AdjacencyMatrixGraph instance, it calls `AdjacencyMatrixGraph.get_neighbors()`.

This is standard OOP polymorphism, not method overloading (which is when you have multiple methods with the same name but different parameter signatures in the same class).

Both classes inherit from the abstract Graph base class:
```
pythonclass Graph(ABC):
    @abstractmethod
    def get_neighbors(self, node_id: int) -> List[Tuple[int, float]]:
        """Return list of (neighbor_id, weight) tuples."""
        pass

class AdjacencyListGraph(Graph):
    def get_neighbors(self, node_id: int) -> List[Tuple[int, float]]:
        return self.adj_list[node_id]

class AdjacencyMatrixGraph(Graph):
    def get_neighbors(self, node_id: int) -> List[Tuple[int, float]]:
        # ... scan entire row ...
The dispatch happens based on which class you instantiate:
python# In load_graph():
if use_matrix:
    graph = AdjacencyMatrixGraph(node_count)
else:
    graph = AdjacencyListGraph()
```

# Later in dijkstra():
```
for neighbor, weight in graph.get_neighbors(current):  # calls the right one
```

Python resolves graph.get_neighbors() at runtime based on the object's actual type, not the variable's declared type. 

So when graph is an AdjacencyListGraph instance, it calls that class's implementation.