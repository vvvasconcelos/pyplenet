"""
Network statistics computation module.

Functions
---------
calculate_reciprocity : Compute reciprocity coefficient for directed graphs
calculate_clustering_coefficient : Compute average clustering coefficient using sampling
shortest_path_sample : Calculate shortest path lengths using BFS sampling
degree_distribution : Extract degree distributions efficiently
runstats : Comprehensive network analysis with optional visualizations

Examples
--------
>>> from graph import FileBasedGraph
>>> from netstats import runstats
>>> graph = FileBasedGraph("my_network")
>>> runstats(graph, sample_size=1000)
=== Network Statistics ===
Number of nodes: 10000
Number of edges: 50000
...
"""
import os
import random
from collections import defaultdict, deque

import numpy as np
import matplotlib.pyplot as plt

def calculate_reciprocity(G):
    """
    Calculate reciprocity coefficient for a directed graph.
    
    Reciprocity measures the tendency of vertex pairs to form mutual connections.
    It is defined as the proportion of edges that have a reciprocal edge in the
    opposite direction.
    
    Parameters
    ----------
    G : FileBasedGraph or InMemoryGraph
        A directed graph object. Must have is_directed(), number_of_edges(),
        and adjacency access methods.
        
    Returns
    -------
    float or None
        Reciprocity coefficient (0.0 to 1.0), where:
        - 0.0: No reciprocal edges
        - 1.0: All edges are reciprocal
        - None: Graph is undirected
        
    Notes
    -----
    For file-based graphs, this function reads the adjacency file twice:
    once to collect all edges, and once to count reciprocal pairs. This
    approach minimizes memory usage at the cost of additional I/O.
    
    The reciprocity is calculated as:
    reciprocity = (number of reciprocal edge pairs) / (total number of edges)
    
    Examples
    --------
    >>> reciprocity = calculate_reciprocity(graph)
    >>> print(f"Graph reciprocity: {reciprocity:.3f}")
    Graph reciprocity: 0.234
    """
    if not G.is_directed():
        return None
    
    total_edges = G.number_of_edges()
    if total_edges == 0:
        return 0
    
    reciprocal_edges = 0
    
    # For file-based graphs, we need to be efficient
    if hasattr(G, 'adjacency_file'):
        # File-based implementation
        edge_set = set()
        
        # First pass: collect all edges
        if os.path.exists(G.adjacency_file):
            with open(G.adjacency_file, 'rb') as f:
                import struct
                while True:
                    data = f.read(80000)  # Read in chunks
                    if not data:
                        break
                    num_edges = len(data) // 8
                    edges = struct.unpack(f'{num_edges * 2}I', data)
                    for i in range(0, len(edges), 2):
                        src, dst = edges[i], edges[i + 1]
                        edge_set.add((src, dst))
        
        # Second pass: count reciprocal edges
        for src, dst in edge_set:
            if (dst, src) in edge_set:
                reciprocal_edges += 1
        
        # Each reciprocal pair is counted twice, so divide by 2
        reciprocal_edges //= 2
        
    else:
        # In-memory implementation
        for src in range(G.number_of_nodes()):
            out_neighbors = G.get_out_edges(src)
            for dst in out_neighbors:
                if src in G.get_in_edges(dst):
                    reciprocal_edges += 1
        reciprocal_edges //= 2
    
    return reciprocal_edges / total_edges if total_edges > 0 else 0

def calculate_clustering_coefficient(G, sample_size=1000):
    """
    Calculate average clustering coefficient using node sampling.
    
    The clustering coefficient measures the degree to which nodes in a graph
    tend to cluster together. For large graphs, this function uses sampling
    to compute an estimate efficiently.
    
    Parameters
    ----------
    G : FileBasedGraph or InMemoryGraph
        Graph object with neighbor access methods
    sample_size : int, optional
        Maximum number of nodes to sample for computation. Default is 1000.
        Actual sample size is min(sample_size, total_nodes).
        
    Returns
    -------
    float
        Average clustering coefficient (0.0 to 1.0), where:
        - 0.0: No clustering (no triangles)
        - 1.0: Perfect clustering (complete local neighborhoods)
        
    Notes
    -----
    The graph is treated as undirected for clustering computation, considering
    both incoming and outgoing edges as neighbors. Nodes with fewer than 2
    neighbors are excluded from the calculation.
    
    For each sampled node, the clustering coefficient is:
    C_i = (number of edges between neighbors) / (possible edges between neighbors)
    
    The function returns the average across all sampled nodes with sufficient
    neighbors.
    
    Examples
    --------
    >>> clustering = calculate_clustering_coefficient(graph, sample_size=500)
    >>> print(f"Average clustering coefficient: {clustering:.4f}")
    Average clustering coefficient: 0.1234
    """
    if G.number_of_nodes() == 0:
        return 0
    
    # Sample nodes for efficiency
    nodes_to_sample = min(sample_size, G.number_of_nodes())
    sampled_nodes = random.sample(list(range(G.number_of_nodes())), nodes_to_sample)
    
    clustering_coeffs = []
    
    for node in sampled_nodes:
        # Get neighbors (treating as undirected for clustering)
        neighbors = set()
        neighbors.update(G.get_out_edges(node))
        neighbors.update(G.get_in_edges(node))
        neighbors.discard(node)  # Remove self if present
        
        if len(neighbors) < 2:
            continue
        
        # Count edges between neighbors
        edges_between_neighbors = 0
        neighbor_list = list(neighbors)
        
        for i, neighbor1 in enumerate(neighbor_list):
            for neighbor2 in neighbor_list[i+1:]:
                # Check if edge exists in either direction (undirected)
                if (neighbor2 in G.get_out_edges(neighbor1) or 
                    neighbor1 in G.get_out_edges(neighbor2)):
                    edges_between_neighbors += 1
        
        # Clustering coefficient for this node
        possible_edges = len(neighbors) * (len(neighbors) - 1) // 2
        if possible_edges > 0:
            clustering_coeffs.append(edges_between_neighbors / possible_edges)
    
    return np.mean(clustering_coeffs) if clustering_coeffs else 0

def shortest_path_sample(G, sample_size=50, max_targets_per_source=100):
    """
    Calculate shortest path lengths using BFS sampling.
    
    Computes shortest path lengths between sampled node pairs using breadth-first
    search. This provides an estimate of the graph's path length distribution
    without computing all-pairs shortest paths.
    
    Parameters
    ----------
    G : FileBasedGraph or InMemoryGraph
        Graph object with outgoing edge access methods
    sample_size : int, optional
        Number of source nodes to sample. Default is 50.
    max_targets_per_source : int, optional
        Maximum number of target nodes per source. Default is 100.
        Prevents excessive computation for highly connected graphs.
        
    Returns
    -------
    list of int
        List of shortest path lengths from sampled source-target pairs.
        Empty list if no paths found.
        
    Notes
    -----
    The function uses BFS to explore reachable nodes from each sampled source.
    If a source can reach more than max_targets_per_source nodes, targets
    are randomly sampled to limit computation time.
    
    Path lengths exclude self-loops (distance 0 from node to itself).
    
    Examples
    --------
    >>> path_lengths = shortest_path_sample(graph, sample_size=100)
    >>> avg_path_length = np.mean(path_lengths) if path_lengths else 0
    >>> print(f"Average path length: {avg_path_length:.2f}")
    Average path length: 3.45
    """
    if G.number_of_nodes() == 0:
        return []
    
    sample_size = min(sample_size, G.number_of_nodes())
    sampled_nodes = random.sample(list(range(G.number_of_nodes())), sample_size)
    
    path_lengths = []
    
    for source in sampled_nodes:
        # BFS from source
        distances = {source: 0}
        queue = deque([source])
        
        while queue:
            current = queue.popleft()
            current_dist = distances[current]
            
            # Explore outgoing edges
            for neighbor in G.get_out_edges(current):
                if neighbor not in distances:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        
        # Collect distances (excluding self)
        targets = [node for node in distances if node != source]
        if len(targets) > max_targets_per_source:
            targets = random.sample(targets, max_targets_per_source)
        
        path_lengths.extend([distances[target] for target in targets])
    
    return path_lengths

def degree_distribution(G):
    """
    Calculate in-degree and out-degree distributions efficiently.
    
    Extracts degree information for all nodes in the graph, excluding
    isolated nodes (degree 0) from the returned distributions.
    
    Parameters
    ----------
    G : FileBasedGraph or InMemoryGraph
        Graph object with degree computation methods
        
    Returns
    -------
    in_degrees : list of int
        List of in-degrees for non-isolated nodes
    out_degrees : list of int
        List of out-degrees for non-isolated nodes
        
    Notes
    -----
    Isolated nodes (nodes with both in-degree and out-degree of 0) are
    excluded to focus on the connected component structure. For analysis
    including isolated nodes, use G.in_degree() and G.out_degree() directly.
    
    Examples
    --------
    >>> in_degrees, out_degrees = degree_distribution(graph)
    >>> print(f"Max out-degree: {max(out_degrees) if out_degrees else 0}")
    >>> print(f"Average in-degree: {np.mean(in_degrees) if in_degrees else 0:.2f}")
    Max out-degree: 45
    Average in-degree: 12.34
    """
    in_degrees = []
    out_degrees = []
    
    # Get degree information
    in_degree_data = G.in_degree()
    out_degree_data = G.out_degree()
    
    for node_id, degree in in_degree_data:
        if degree > 0:  # Exclude isolates
            in_degrees.append(degree)
    
    for node_id, degree in out_degree_data:
        if degree > 0:  # Exclude isolates
            out_degrees.append(degree)
    
    return in_degrees, out_degrees

def runstats(G, show_plots=True, sample_size=100):
    """
    Run comprehensive network statistics analysis on a graph.
    
    Performs a complete statistical analysis of the network including
    basic properties, reciprocity, clustering, path lengths, and degree
    distributions. Optimized for large networks using sampling techniques.
    
    Parameters
    ----------
    G : FileBasedGraph or InMemoryGraph
        Graph object to analyze
    show_plots : bool, optional
        Whether to display matplotlib plots of distributions. Default is True.
    sample_size : int, optional
        Sample size for computationally expensive metrics. Default is 100.
        
    Notes
    -----
    The function automatically adapts computation strategies based on graph
    size and type. For very large graphs, sampling is used to estimate:
    - Clustering coefficient (sampled nodes)
    - Shortest path lengths (sampled node pairs)
    
    Degree distributions and reciprocity are computed exactly for all nodes.
    
    Output includes:
    - Basic graph properties (nodes, edges, directedness)
    - Reciprocity coefficient (directed graphs only)
    - Average clustering coefficient
    - Shortest path length distribution and statistics
    - Degree distribution plots and statistics
    
    Examples
    --------
    >>> runstats(graph, show_plots=True, sample_size=1000)
    === Network Statistics ===
    Number of nodes: 50000
    Number of edges: 250000
    Is directed: True
    
    Calculating reciprocity...
    Reciprocity: 0.1234
    ...
    
    >>> runstats(graph, show_plots=False, sample_size=500)  # No plots
    """
    print("=== Network Statistics ===")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Is directed: {G.is_directed()}")
    print()
    
    # Reciprocity
    if G.is_directed():
        print("Calculating reciprocity...")
        reciprocity = calculate_reciprocity(G)
        print(f"Reciprocity: {reciprocity:.4f}")
    else:
        print("Reciprocity: Not applicable (graph is undirected)")
    
    # Clustering coefficient
    print("Calculating clustering coefficient...")
    avg_clustering = calculate_clustering_coefficient(G, sample_size)
    print(f"Average clustering coefficient: {avg_clustering:.4f}")
    
    # Shortest path lengths
    print("Calculating shortest path distribution...")
    path_lengths = shortest_path_sample(G, sample_size)
    
    if path_lengths and show_plots:
        plt.figure(figsize=(8, 5))
        plt.hist(path_lengths, bins=range(min(path_lengths), max(path_lengths)+2), 
                align='left', rwidth=0.8)
        plt.title("Shortest Path Length Distribution (Sample)")
        plt.xlabel("Path Length")
        plt.ylabel("Frequency")
        plt.show()
        print(f"Average path length: {np.mean(path_lengths):.2f}")
        print(f"Path length std: {np.std(path_lengths):.2f}")
    elif path_lengths:
        print(f"Average path length: {np.mean(path_lengths):.2f}")
        print(f"Path length std: {np.std(path_lengths):.2f}")
    else:
        print("No paths found in the sampled nodes.")
    
    # Degree statistics
    print("Calculating degree distribution...")
    in_degrees, out_degrees = degree_distribution(G)
    
    if out_degrees:
        avg_out_degree = np.mean(out_degrees)
        print(f"Average out-degree (excluding isolates): {avg_out_degree:.2f}")
        
        if show_plots:
            plt.figure(figsize=(12, 4))
            
            # Out-degree distribution
            plt.subplot(1, 2, 1)
            plt.hist(out_degrees, bins=min(50, max(out_degrees)), alpha=0.7)
            plt.title("Out-Degree Distribution")
            plt.xlabel("Out-Degree")
            plt.ylabel("Number of Nodes")
            
            # In-degree distribution
            plt.subplot(1, 2, 2)
            plt.hist(in_degrees, bins=min(50, max(in_degrees)), alpha=0.7)
            plt.title("In-Degree Distribution")
            plt.xlabel("In-Degree")
            plt.ylabel("Number of Nodes")
            
            plt.tight_layout()
            plt.show()
    else:
        print("No non-isolated nodes found.")
    
    if in_degrees:
        avg_in_degree = np.mean(in_degrees)
        print(f"Average in-degree (excluding isolates): {avg_in_degree:.2f}")
    
    print("=== Statistics Complete ===")

if __name__ == "__main__":
    """
    Example usage and testing of network statistics functions.
    
    This section demonstrates how to load a graph and run comprehensive
    statistics analysis. Modify the graph loading logic as needed for
    your specific graph format and file paths.
    """
    # Import graph class - update path as needed
    from pyplenet.core.graph import FileBasedGraph
    
    try:
        # Example: Load a file-based graph
        graph = FileBasedGraph('graph_data')
        if graph.number_of_nodes() > 0:
            print("Loaded graph from file-based storage")
            runstats(graph, show_plots=True, sample_size=500)
        else:
            print("No graph data found in 'graph_data' directory")
            print("Generate a graph first using generate.py")
            
    except FileNotFoundError:
        print("Graph data directory not found.")
        print("Run generate.py first to create a network.")
    except Exception as e:
        print(f"Error loading graph: {e}")
        print("Make sure the graph files are properly formatted and accessible.")
