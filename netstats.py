import networkx as nx
import matplotlib.pyplot as plt
import pickle 
import random

def runstats(G):
    # Basic stats
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())
    print("Is directed:", G.is_directed())

    # Reciprocity
    if G.is_directed():
        reciprocity = nx.reciprocity(G)
        print("Reciprocity:", reciprocity)
        # Clustering coefficient for directed graphs (convert to undirected for clustering)
        avg_clustering = nx.average_clustering(G.to_undirected())
        print("Average clustering coefficient (undirected projection):", avg_clustering)
    else:
        print("Reciprocity: Not applicable (graph is undirected)")
        # Clustering coefficient for undirected graphs
        avg_clustering = nx.average_clustering(G)
        print("Average clustering coefficient:", avg_clustering)
    # Shortest path length distribution from random sample of nodes

    sample_size = min(50, G.number_of_nodes())
    nodes = list(G.nodes())
    sample_nodes = random.sample(nodes, sample_size)

    max_targets_per_source = 100  # Limit for each source node

    path_lengths = []
    for node in sample_nodes:
        lengths = nx.single_source_shortest_path_length(G, node)
        # Exclude self (distance 0)
        targets = [t for t in lengths if t != node]
        if len(targets) > max_targets_per_source:
            targets = random.sample(targets, max_targets_per_source)
        path_lengths.extend([lengths[t] for t in targets])

    if path_lengths:
        plt.figure(figsize=(8, 5))
        plt.hist(path_lengths, bins=range(min(path_lengths), max(path_lengths)+2), align='left', rwidth=0.8)
        plt.title("Shortest Path Length Distribution (Sample of Nodes)")
        plt.xlabel("Path Length")
        plt.ylabel("Frequency")
        plt.show()
    else:
        print("No paths found in the sampled nodes.")
    
    # Degree statistics (excluding isolates)
    degrees = [d for n, d in G.in_degree() if d > 0]
    if degrees:
        avg_degree = sum(degrees) / len(degrees)
        print("Average degree (excluding isolates):", avg_degree)

        plt.figure(figsize=(8, 5))
        plt.hist(degrees, bins=range(min(degrees), max(degrees)+2), align='left', rwidth=0.8)
        plt.title("Degree Distribution")
        plt.xlabel("Degree")
        plt.ylabel("Number of Nodes")
        plt.show()
    else:
        print("No non-isolated nodes found.")
        
if __name__ == "__main__":
    # Read the graph from file
    with open('generated_graph.gpickle', 'rb') as f:
        G = pickle.load(f)
    runstats(G)