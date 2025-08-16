#!/usr/bin/env python3
"""
Demo script for PyPleNet: Generate network, extract subgraph, and visualize.
"""

import numpy as np
import tempfile
import shutil
import os

# Import PyPleNet modules
from pyplenet.core.generate import generate
from pyplenet.core.graph import FileBasedGraph

# Network generation parameters
pops = 'Data/fake_tab_n.csv'        # Population groups data
links = 'Data/tab_werkschool.xlsx'  # Interaction/links data
fraction = 1                    # Preferential attachment fraction
scale = 0.5                         # Population scaling (reduced for testing)
reciprocity = 0.2                     # No reciprocal edges in this example

def main(use_existing=False, graph_path="graph_data", center_node_seed = None):
    """
    Main demo function: generate network, extract subgraph, and visualize.
    
    Parameters
    ----------
    use_existing : bool, optional
        If True, load existing graph from graph_path instead of generating new one.
        Default is False.
    graph_path : str, optional
        Path to existing graph directory or where to save new graph.
        Default is "graph_data".
    """
    print("="*60)
    print("PyPleNet Demo: Network Generation and Visualization")
    print("="*60)
    
    # Step 1: Load or generate the full network
    if use_existing and os.path.exists(graph_path):
        print(f"\nStep 1: Loading existing FileBasedGraph from '{graph_path}'...")
        try:
            graph = FileBasedGraph(graph_path)
            print(f"✓ Existing network loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        except Exception as e:
            print(f"✗ Failed to load existing graph: {e}")
            print("Falling back to generating new graph...")
            use_existing = False
    else:
        if use_existing:
            print(f"\nGraph path '{graph_path}' not found. Generating new graph...")
        use_existing = False
    
    if not use_existing:
        print(f"\nStep 1: Generating new FileBasedGraph...")
        print(f"Population file: {pops}")
        print(f"Links file: {links}")
        print(f"Parameters: fraction={fraction}, scale={scale}, reciprocity={reciprocity}")
        
        # Generate the network
        graph = generate(pops, links, fraction, scale, reciprocity, base_path=graph_path)
        
        print(f"✓ Network generated: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Step 2: Extract a subgraph
    print("\nStep 2: Extracting subgraph...")
    
    # Choose a random center node
    np.random.seed(center_node_seed)  # For reproducible results
    center_node = np.random.randint(0, graph.number_of_nodes())
    max_nodes = 100
    
    print(f"Center node: {center_node}")
    print(f"Max nodes to extract: {max_nodes}")
    
    # Create temporary directory for subgraph
    subgraph_temp_dir = tempfile.mkdtemp(prefix="demo_subgraph_")
    
    try:
        # Extract subgraph
        subgraph = graph.extract_subgraph(
            center_node=center_node,
            max_nodes=max_nodes,
            output_path=subgraph_temp_dir
        )
        
        print(f"✓ Subgraph extracted: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
        
        # Step 3: Convert to NetworkX and plot
        print("\nStep 3: Converting to NetworkX and creating visualization...")
        
        try:
            # Convert to NetworkX
            nx_graph = subgraph.to_networkx()
            print(f"✓ NetworkX conversion successful: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges")
            
            # Plot the graph
            try:
                import matplotlib.pyplot as plt
                import networkx as nx
                
                print("Creating network visualization...")
                
                # Create the plot
                plt.figure(figsize=(12, 8))
                
                # Use a tree layout (hierarchical) with center_node as root
                try:
                    pos = nx.multipartite_layout(nx_graph, subset_key=None)
                except Exception:
                    # Fallback to spring layout if multipartite_layout fails
                    pos = nx.spring_layout(nx_graph, k=1, iterations=50)
                # Optionally, re-center the root node
                if center_node in pos:
                    pos[center_node] = np.array([0.5, 1.0])
                node_size = 300
                
                # Color nodes by group if available
                node_colors = []
                if nx_graph.nodes() and 'group' in next(iter(nx_graph.nodes(data=True)))[1]:
                    # Get unique groups and assign colors
                    groups = set()
                    for _, attrs in nx_graph.nodes(data=True):
                        groups.add(attrs.get('group', 'unknown'))
                    
                    group_colors = {}
                    colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
                    for i, group in enumerate(groups):
                        group_colors[group] = colors[i]
                    
                    for node, attrs in nx_graph.nodes(data=True):
                        group = attrs.get('group', 'unknown')
                        node_colors.append(group_colors[group])
                    
                    print(f"Found {len(groups)} different groups in subgraph")
                else:
                    # Default coloring
                    node_colors = ['lightblue'] * nx_graph.number_of_nodes()
                
                # Highlight center node
                center_color = 'red'
                if center_node in nx_graph.nodes():
                    center_idx = list(nx_graph.nodes()).index(center_node)
                    node_colors[center_idx] = center_color
                
                # Draw the graph
                nx.draw(nx_graph, pos, 
                       node_color=node_colors,
                       node_size=node_size,
                       with_labels=True if nx_graph.number_of_nodes() <= 30 else False,
                       font_size=8,
                       font_color='black',
                       edge_color='gray',
                       alpha=0.8,
                       arrows=True,
                       arrowsize=10,
                       arrowstyle='->')
                
                # Add title and legend
                plt.title(f'PyPleNet Subgraph Visualization\n'
                         f'{nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges\n'
                         f'Center node: {center_node} (red)', 
                         fontsize=14, pad=20)
                
                # Add legend for groups if available
                if len(set(node_colors)) > 2:  # More than just default + center
                    legend_elements = []
                    if 'group_colors' in locals():
                        for group, color in group_colors.items():
                            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                           markerfacecolor=color, markersize=10, 
                                                           label=f'Group {group}'))
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                   markerfacecolor=center_color, markersize=10, 
                                                   label='Center node'))
                    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
                
                plt.tight_layout()
                
                # Save the plot
                output_file = 'demo_network_visualization.png'
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"✓ Visualization saved as '{output_file}'")
                
                # Show the plot
                plt.show()
                
                # Step 4: Print some network statistics
                print("\nStep 4: Network Analysis...")
                
                print(f"Graph Statistics:")
                print(f"  - Nodes: {nx_graph.number_of_nodes()}")
                print(f"  - Edges: {nx_graph.number_of_edges()}")
                print(f"  - Density: {nx.density(nx_graph):.4f}")
                print(f"  - Is strongly connected: {nx.is_strongly_connected(nx_graph)}")
                print(f"  - Number of strongly connected components: {nx.number_strongly_connected_components(nx_graph)}")
                
                # Degree statistics
                degrees = [d for n, d in nx_graph.degree()]
                if degrees:
                    print(f"  - Average degree: {np.mean(degrees):.2f}")
                    print(f"  - Max degree: {np.max(degrees)}")
                    print(f"  - Min degree: {np.min(degrees)}")
                
                # Center node statistics
                if center_node in nx_graph.nodes():
                    center_degree = nx_graph.degree(center_node)
                    print(f"  - Center node degree: {center_degree}")
                
                # Top nodes by degree
                degree_centrality = nx.degree_centrality(nx_graph)
                top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"  - Top 5 nodes by degree centrality:")
                for node, centrality in top_nodes:
                    print(f"    Node {node}: {centrality:.3f}")
                
            except ImportError as e:
                print(f"Matplotlib not available for plotting: {e}")
                print("Install with: pip install matplotlib")
                print("NetworkX graph created successfully but cannot display plot.")
                
        except ImportError as e:
            print(f"NetworkX not available: {e}")
            print("Install with: pip install networkx")
            
    finally:
        # Cleanup temporary subgraph directory
        if os.path.exists(subgraph_temp_dir):
            shutil.rmtree(subgraph_temp_dir)
            print(f"✓ Cleaned up temporary files")
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)

if __name__ == "__main__":
    #import argparse
    
    #parser = argparse.ArgumentParser(description='PyPleNet Demo: Generate network, extract subgraph, and visualize')
    #parser.add_argument('--use-existing', action='store_true', 
                       #help='Use existing graph instead of generating new one')
    #parser.add_argument('--graph-path', default='graph_data', 
                       #help='Path to graph directory (default: graph_data)')
    
    #args = parser.parse_args()
    
    main()
    