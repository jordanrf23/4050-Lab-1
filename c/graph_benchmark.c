/*
 * CS3050 Lab 6 - Graph Representations in C
 * 
 * This implementation provides both adjacency list and adjacency matrix
 * representations to demonstrate their performance characteristics.
 * 
 * Compile: make
 * Run: ./graph_benchmark ../data/medium_nodes.csv ../data/medium_edges.csv
 */

#define _POSIX_C_SOURCE 199309L
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <float.h>

#define MAX_LINE_LENGTH 256
#define INITIAL_CAPACITY 100
#define EARTH_RADIUS_KM 6371.0

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================================
 * Data Structures
 * ============================================================================ */

typedef struct {
    int id;
    double lat;
    double lon;
} Node;

typedef struct EdgeNode {
    int to;
    double weight;
    struct EdgeNode* next;
} EdgeNode;

/* Adjacency List Graph */
typedef struct {
    Node* nodes;
    EdgeNode** adj_list;  /* Array of linked lists */
    int num_nodes;
    int num_edges;
    int capacity;
} AdjListGraph;

/* Adjacency Matrix Graph */
typedef struct {
    Node* nodes;
    double** matrix;      /* 2D array */
    int num_nodes;
    int num_edges;
    int capacity;
} AdjMatrixGraph;

/* Priority Queue Entry for Dijkstra */
typedef struct {
    double priority;
    int node_id;
} PQEntry;

typedef struct {
    PQEntry* heap;
    int size;
    int capacity;
} PriorityQueue;

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

double haversine_distance(double lat1, double lon1, double lat2, double lon2) {
    double phi1 = lat1 * M_PI / 180.0;
    double phi2 = lat2 * M_PI / 180.0;
    double delta_phi = (lat2 - lat1) * M_PI / 180.0;
    double delta_lambda = (lon2 - lon1) * M_PI / 180.0;
    
    double a = sin(delta_phi/2) * sin(delta_phi/2) +
               cos(phi1) * cos(phi2) * sin(delta_lambda/2) * sin(delta_lambda/2);
    
    return 2 * EARTH_RADIUS_KM * asin(sqrt(a));
}

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

/* ============================================================================
 * Priority Queue Implementation
 * ============================================================================ */

PriorityQueue* pq_create(int capacity) {
    PriorityQueue* pq = malloc(sizeof(PriorityQueue));
    pq->heap = malloc(capacity * sizeof(PQEntry));
    pq->size = 0;
    pq->capacity = capacity;
    return pq;
}

void pq_destroy(PriorityQueue* pq) {
    free(pq->heap);
    free(pq);
}

void pq_push(PriorityQueue* pq, double priority, int node_id) {
    if (pq->size >= pq->capacity) {
        pq->capacity *= 2;
        pq->heap = realloc(pq->heap, pq->capacity * sizeof(PQEntry));
    }
    
    int i = pq->size++;
    pq->heap[i].priority = priority;
    pq->heap[i].node_id = node_id;
    
    /* Bubble up */
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (pq->heap[parent].priority <= pq->heap[i].priority) break;
        PQEntry temp = pq->heap[parent];
        pq->heap[parent] = pq->heap[i];
        pq->heap[i] = temp;
        i = parent;
    }
}

PQEntry pq_pop(PriorityQueue* pq) {
    PQEntry min = pq->heap[0];
    pq->heap[0] = pq->heap[--pq->size];
    
    /* Bubble down */
    int i = 0;
    while (2*i + 1 < pq->size) {
        int left = 2*i + 1;
        int right = 2*i + 2;
        int smallest = i;
        
        if (pq->heap[left].priority < pq->heap[smallest].priority)
            smallest = left;
        if (right < pq->size && pq->heap[right].priority < pq->heap[smallest].priority)
            smallest = right;
        
        if (smallest == i) break;
        
        PQEntry temp = pq->heap[smallest];
        pq->heap[smallest] = pq->heap[i];
        pq->heap[i] = temp;
        i = smallest;
    }
    
    return min;
}

bool pq_empty(PriorityQueue* pq) {
    return pq->size == 0;
}

/* ============================================================================
 * Adjacency List Implementation
 * ============================================================================ */

AdjListGraph* adjlist_create(int initial_capacity) {
    AdjListGraph* g = malloc(sizeof(AdjListGraph));
    g->nodes = malloc(initial_capacity * sizeof(Node));
    g->adj_list = calloc(initial_capacity, sizeof(EdgeNode*));
    g->num_nodes = 0;
    g->num_edges = 0;
    g->capacity = initial_capacity;
    return g;
}

void adjlist_destroy(AdjListGraph* g) {
    for (int i = 0; i < g->capacity; i++) {
        EdgeNode* current = g->adj_list[i];
        while (current) {
            EdgeNode* next = current->next;
            free(current);
            current = next;
        }
    }
    free(g->adj_list);
    free(g->nodes);
    free(g);
}

void adjlist_ensure_capacity(AdjListGraph* g, int min_capacity) {
    if (min_capacity <= g->capacity) return;
    
    int new_capacity = g->capacity;
    while (new_capacity < min_capacity) new_capacity *= 2;
    
    g->nodes = realloc(g->nodes, new_capacity * sizeof(Node));
    g->adj_list = realloc(g->adj_list, new_capacity * sizeof(EdgeNode*));
    
    for (int i = g->capacity; i < new_capacity; i++) {
        g->adj_list[i] = NULL;
    }
    
    g->capacity = new_capacity;
}

void adjlist_add_node(AdjListGraph* g, int id, double lat, double lon) {
    adjlist_ensure_capacity(g, id + 1);
    g->nodes[id].id = id;
    g->nodes[id].lat = lat;
    g->nodes[id].lon = lon;
    if (id >= g->num_nodes) g->num_nodes = id + 1;
}

void adjlist_add_edge(AdjListGraph* g, int from, int to, double weight) {
    /* Add edge from -> to */
    EdgeNode* edge1 = malloc(sizeof(EdgeNode));
    edge1->to = to;
    edge1->weight = weight;
    edge1->next = g->adj_list[from];
    g->adj_list[from] = edge1;
    
    /* Add edge to -> from (undirected) */
    EdgeNode* edge2 = malloc(sizeof(EdgeNode));
    edge2->to = from;
    edge2->weight = weight;
    edge2->next = g->adj_list[to];
    g->adj_list[to] = edge2;
    
    g->num_edges++;
}

bool adjlist_has_edge(AdjListGraph* g, int from, int to) {
    EdgeNode* current = g->adj_list[from];
    while (current) {
        if (current->to == to) return true;
        current = current->next;
    }
    return false;
}

double adjlist_get_weight(AdjListGraph* g, int from, int to) {
    EdgeNode* current = g->adj_list[from];
    while (current) {
        if (current->to == to) return current->weight;
        current = current->next;
    }
    return -1.0;
}

size_t adjlist_memory_bytes(AdjListGraph* g) {
    size_t bytes = sizeof(AdjListGraph);
    bytes += g->capacity * sizeof(Node);
    bytes += g->capacity * sizeof(EdgeNode*);
    bytes += g->num_edges * 2 * sizeof(EdgeNode);  /* Each edge stored twice */
    return bytes;
}

/* ============================================================================
 * Adjacency Matrix Implementation
 * ============================================================================ */

AdjMatrixGraph* adjmatrix_create(int num_nodes) {
    printf("  Allocating %dx%d matrix (%.2f GB)...\n", 
           num_nodes, num_nodes, 
           (double)(num_nodes * num_nodes * sizeof(double)) / (1024*1024*1024));
    
    AdjMatrixGraph* g = malloc(sizeof(AdjMatrixGraph));
    g->nodes = malloc(num_nodes * sizeof(Node));
    g->matrix = malloc(num_nodes * sizeof(double*));
    
    for (int i = 0; i < num_nodes; i++) {
        g->matrix[i] = calloc(num_nodes, sizeof(double));
        if (!g->matrix[i]) {
            fprintf(stderr, "ERROR: Failed to allocate row %d of matrix!\n", i);
            exit(1);
        }
    }
    
    g->num_nodes = 0;
    g->num_edges = 0;
    g->capacity = num_nodes;
    
    printf("  Matrix allocated.\n");
    return g;
}

void adjmatrix_destroy(AdjMatrixGraph* g) {
    for (int i = 0; i < g->capacity; i++) {
        free(g->matrix[i]);
    }
    free(g->matrix);
    free(g->nodes);
    free(g);
}

void adjmatrix_add_node(AdjMatrixGraph* g, int id, double lat, double lon) {
    g->nodes[id].id = id;
    g->nodes[id].lat = lat;
    g->nodes[id].lon = lon;
    if (id >= g->num_nodes) g->num_nodes = id + 1;
}

void adjmatrix_add_edge(AdjMatrixGraph* g, int from, int to, double weight) {
    g->matrix[from][to] = weight;
    g->matrix[to][from] = weight;
    g->num_edges++;
}

bool adjmatrix_has_edge(AdjMatrixGraph* g, int from, int to) {
    return g->matrix[from][to] > 0;
}

double adjmatrix_get_weight(AdjMatrixGraph* g, int from, int to) {
    return g->matrix[from][to];
}

size_t adjmatrix_memory_bytes(AdjMatrixGraph* g) {
    size_t bytes = sizeof(AdjMatrixGraph);
    bytes += g->capacity * sizeof(Node);
    bytes += g->capacity * sizeof(double*);
    bytes += (size_t)g->capacity * g->capacity * sizeof(double);
    return bytes;
}

/* ============================================================================
 * File Loading
 * ============================================================================ */

int count_lines(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) return 0;
    
    int count = 0;
    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), f)) count++;
    
    fclose(f);
    return count - 1;  /* Subtract header */
}

void load_adjlist_graph(AdjListGraph* g, const char* nodes_file, const char* edges_file) {
    FILE* f;
    char line[MAX_LINE_LENGTH];
    
    /* Load nodes */
    f = fopen(nodes_file, "r");
    if (!f) { perror("Cannot open nodes file"); exit(1); }
    
    fgets(line, sizeof(line), f);  /* Skip header */
    while (fgets(line, sizeof(line), f)) {
        int id;
        double lat, lon;
        sscanf(line, "%d,%lf,%lf", &id, &lat, &lon);
        adjlist_add_node(g, id, lat, lon);
    }
    fclose(f);
    
    /* Load edges */
    f = fopen(edges_file, "r");
    if (!f) { perror("Cannot open edges file"); exit(1); }
    
    fgets(line, sizeof(line), f);  /* Skip header */
    while (fgets(line, sizeof(line), f)) {
        int from, to;
        double weight;
        sscanf(line, "%d,%d,%lf", &from, &to, &weight);
        adjlist_add_edge(g, from, to, weight);
    }
    fclose(f);
}

void load_adjmatrix_graph(AdjMatrixGraph* g, const char* nodes_file, const char* edges_file) {
    FILE* f;
    char line[MAX_LINE_LENGTH];
    
    /* Load nodes */
    f = fopen(nodes_file, "r");
    if (!f) { perror("Cannot open nodes file"); exit(1); }
    
    fgets(line, sizeof(line), f);  /* Skip header */
    while (fgets(line, sizeof(line), f)) {
        int id;
        double lat, lon;
        sscanf(line, "%d,%lf,%lf", &id, &lat, &lon);
        adjmatrix_add_node(g, id, lat, lon);
    }
    fclose(f);
    
    /* Load edges */
    f = fopen(edges_file, "r");
    if (!f) { perror("Cannot open edges file"); exit(1); }
    
    fgets(line, sizeof(line), f);  /* Skip header */
    while (fgets(line, sizeof(line), f)) {
        int from, to;
        double weight;
        sscanf(line, "%d,%d,%lf", &from, &to, &weight);
        adjmatrix_add_edge(g, from, to, weight);
    }
    fclose(f);
}

/* ============================================================================
 * Dijkstra's Algorithm (for both representations)
 * ============================================================================ */

typedef struct {
    int* path;
    int path_length;
    double distance;
    int nodes_explored;
} PathResult;

PathResult dijkstra_adjlist(AdjListGraph* g, int start, int goal) {
    PathResult result = {NULL, 0, DBL_MAX, 0};
    
    double* dist = malloc(g->num_nodes * sizeof(double));
    int* prev = malloc(g->num_nodes * sizeof(int));
    bool* visited = calloc(g->num_nodes, sizeof(bool));
    
    for (int i = 0; i < g->num_nodes; i++) {
        dist[i] = DBL_MAX;
        prev[i] = -1;
    }
    dist[start] = 0;
    
    PriorityQueue* pq = pq_create(g->num_nodes);
    pq_push(pq, 0, start);
    
    while (!pq_empty(pq)) {
        PQEntry entry = pq_pop(pq);
        int current = entry.node_id;
        
        if (visited[current]) continue;
        visited[current] = true;
        result.nodes_explored++;
        
        if (current == goal) break;
        
        /* Iterate neighbors - this is O(degree) for adjacency list */
        EdgeNode* edge = g->adj_list[current];
        while (edge) {
            if (!visited[edge->to]) {
                double new_dist = dist[current] + edge->weight;
                if (new_dist < dist[edge->to]) {
                    dist[edge->to] = new_dist;
                    prev[edge->to] = current;
                    pq_push(pq, new_dist, edge->to);
                }
            }
            edge = edge->next;
        }
    }
    
    /* Reconstruct path */
    if (prev[goal] != -1 || goal == start) {
        result.distance = dist[goal];
        
        /* Count path length */
        int len = 0;
        int curr = goal;
        while (curr != -1) { len++; curr = prev[curr]; }
        
        result.path = malloc(len * sizeof(int));
        result.path_length = len;
        
        curr = goal;
        for (int i = len - 1; i >= 0; i--) {
            result.path[i] = curr;
            curr = prev[curr];
        }
    }
    
    pq_destroy(pq);
    free(dist);
    free(prev);
    free(visited);
    
    return result;
}

PathResult dijkstra_adjmatrix(AdjMatrixGraph* g, int start, int goal) {
    PathResult result = {NULL, 0, DBL_MAX, 0};
    
    double* dist = malloc(g->num_nodes * sizeof(double));
    int* prev = malloc(g->num_nodes * sizeof(int));
    bool* visited = calloc(g->num_nodes, sizeof(bool));
    
    for (int i = 0; i < g->num_nodes; i++) {
        dist[i] = DBL_MAX;
        prev[i] = -1;
    }
    dist[start] = 0;
    
    PriorityQueue* pq = pq_create(g->num_nodes);
    pq_push(pq, 0, start);
    
    while (!pq_empty(pq)) {
        PQEntry entry = pq_pop(pq);
        int current = entry.node_id;
        
        if (visited[current]) continue;
        visited[current] = true;
        result.nodes_explored++;
        
        if (current == goal) break;
        
        /* Iterate neighbors - this is O(V) for adjacency matrix! */
        for (int neighbor = 0; neighbor < g->num_nodes; neighbor++) {
            double weight = g->matrix[current][neighbor];
            if (weight > 0 && !visited[neighbor]) {
                double new_dist = dist[current] + weight;
                if (new_dist < dist[neighbor]) {
                    dist[neighbor] = new_dist;
                    prev[neighbor] = current;
                    pq_push(pq, new_dist, neighbor);
                }
            }
        }
    }
    
    /* Reconstruct path */
    if (prev[goal] != -1 || goal == start) {
        result.distance = dist[goal];
        
        int len = 0;
        int curr = goal;
        while (curr != -1) { len++; curr = prev[curr]; }
        
        result.path = malloc(len * sizeof(int));
        result.path_length = len;
        
        curr = goal;
        for (int i = len - 1; i >= 0; i--) {
            result.path[i] = curr;
            curr = prev[curr];
        }
    }
    
    pq_destroy(pq);
    free(dist);
    free(prev);
    free(visited);
    
    return result;
}

/* ============================================================================
 * Benchmarking
 * ============================================================================ */

void benchmark_edge_queries_list(AdjListGraph* g, int num_queries) {
    srand(42);
    double start = get_time_ms();
    
    int hits = 0;
    for (int i = 0; i < num_queries; i++) {
        int a = rand() % g->num_nodes;
        int b = rand() % g->num_nodes;
        if (adjlist_has_edge(g, a, b)) hits++;
    }
    
    double elapsed = get_time_ms() - start;
    printf("  Edge queries: %d in %.1f ms (%.0f q/s), density: %.4f\n",
           num_queries, elapsed, num_queries / (elapsed/1000), (double)hits/num_queries);
}

void benchmark_edge_queries_matrix(AdjMatrixGraph* g, int num_queries) {
    srand(42);
    double start = get_time_ms();
    
    int hits = 0;
    for (int i = 0; i < num_queries; i++) {
        int a = rand() % g->num_nodes;
        int b = rand() % g->num_nodes;
        if (adjmatrix_has_edge(g, a, b)) hits++;
    }
    
    double elapsed = get_time_ms() - start;
    printf("  Edge queries: %d in %.1f ms (%.0f q/s), density: %.4f\n",
           num_queries, elapsed, num_queries / (elapsed/1000), (double)hits/num_queries);
}

void benchmark_dijkstra_list(AdjListGraph* g, int num_queries) {
    srand(123);
    double total_time = 0;
    int total_explored = 0;
    
    for (int i = 0; i < num_queries; i++) {
        int start = rand() % g->num_nodes;
        int goal = rand() % g->num_nodes;
        while (goal == start) goal = rand() % g->num_nodes;
        
        double t1 = get_time_ms();
        PathResult result = dijkstra_adjlist(g, start, goal);
        total_time += get_time_ms() - t1;
        total_explored += result.nodes_explored;
        
        if (result.path) free(result.path);
    }
    
    printf("  Dijkstra: %d queries in %.1f ms (avg %.2f ms, %.0f nodes explored)\n",
           num_queries, total_time, total_time/num_queries, (double)total_explored/num_queries);
}

void benchmark_dijkstra_matrix(AdjMatrixGraph* g, int num_queries) {
    srand(123);
    double total_time = 0;
    int total_explored = 0;
    
    for (int i = 0; i < num_queries; i++) {
        int start = rand() % g->num_nodes;
        int goal = rand() % g->num_nodes;
        while (goal == start) goal = rand() % g->num_nodes;
        
        double t1 = get_time_ms();
        PathResult result = dijkstra_adjmatrix(g, start, goal);
        total_time += get_time_ms() - t1;
        total_explored += result.nodes_explored;
        
        if (result.path) free(result.path);
    }
    
    printf("  Dijkstra: %d queries in %.1f ms (avg %.2f ms, %.0f nodes explored)\n",
           num_queries, total_time, total_time/num_queries, (double)total_explored/num_queries);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s <nodes.csv> <edges.csv>\n", argv[0]);
        return 1;
    }
    
    const char* nodes_file = argv[1];
    const char* edges_file = argv[2];
    
    int num_nodes = count_lines(nodes_file);
    double matrix_gb = (double)(num_nodes * num_nodes * sizeof(double)) / (1024*1024*1024);
    
    printf("\n========================================\n");
    printf(" CS3050 Graph Representation Benchmark\n");
    printf("========================================\n");
    printf("Nodes: %d\n", num_nodes);
    printf("Matrix would require: %.2f GB\n", matrix_gb);
    
    bool run_matrix = matrix_gb < 20.0;
    if (!run_matrix) {
        printf("Matrix benchmark SKIPPED (would exceed memory)\n");
    }
    
    /* Adjacency List */
    printf("\n--- Adjacency List ---\n");
    double t1 = get_time_ms();
    AdjListGraph* list_graph = adjlist_create(num_nodes);
    load_adjlist_graph(list_graph, nodes_file, edges_file);
    double list_load_time = get_time_ms() - t1;
    
    size_t list_mem = adjlist_memory_bytes(list_graph);
    printf("Load time: %.1f ms\n", list_load_time);
    printf("Memory: %.2f MB\n", list_mem / (1024.0*1024.0));
    printf("Nodes: %d, Edges: %d\n", list_graph->num_nodes, list_graph->num_edges);
    
    printf("\nBenchmarks:\n");
    benchmark_edge_queries_list(list_graph, 10000);
    benchmark_dijkstra_list(list_graph, 5);
    
    adjlist_destroy(list_graph);
    
    /* Adjacency Matrix */
    if (run_matrix) {
        printf("\n--- Adjacency Matrix ---\n");
        t1 = get_time_ms();
        AdjMatrixGraph* matrix_graph = adjmatrix_create(num_nodes);
        load_adjmatrix_graph(matrix_graph, nodes_file, edges_file);
        double matrix_load_time = get_time_ms() - t1;
        
        size_t matrix_mem = adjmatrix_memory_bytes(matrix_graph);
        printf("Load time: %.1f ms\n", matrix_load_time);
        printf("Memory: %.2f MB\n", matrix_mem / (1024.0*1024.0));
        
        printf("\nBenchmarks:\n");
        benchmark_edge_queries_matrix(matrix_graph, 10000);
        benchmark_dijkstra_matrix(matrix_graph, 5);
        
        adjmatrix_destroy(matrix_graph);
    }
    
    printf("\n========================================\n");
    printf(" Benchmark complete!\n");
    printf("========================================\n");
    
    return 0;
}
