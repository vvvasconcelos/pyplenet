import matplotlib.pyplot as plt
import random
import numpy as np
from collections import defaultdict, deque
import os

def calculate_reciprocity(G):
    """Calculate reciprocity for a directed graph."""
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
    """Calculate average clustering coefficient using sampling."""
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
    """Calculate shortest path lengths using BFS sampling."""
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
    """Calculate degree distribution efficiently."""
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
    Run comprehensive network statistics on adjacency list graph.
    Optimized for large networks using sampling where necessary.
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
    # Example usage
    from graph import InMemoryGraph
    
    # Load graph from file if it exists
    try:
        graph = InMemoryGraph.load_from_file('generated_graph.pkl')
        print("Loaded graph from file")
        runstats(graph)
    except FileNotFoundError:
        print("No graph file found. Run adjacency_generate.py first.")
    except Exception as e:
        print(f"Error loading graph: {e}")
        print("Run adjacency_generate.py first.")
