from utils import find_nodes, read_file, desc_groups, export_group_to_attrs_csv, export_group_to_nodes_csv
from adjacency_grn import connect_nodes
from adjacency_netstats import runstats
from adjacency_graph import FileBasedGraph, InMemoryGraph
from graph_export import export_all_formats, export_binary_edges, export_edge_list_txt, export_csv_edges
import numpy as np
import math

def init_nodes(G, pops_path, scale):
    """Initialize nodes in the graph based on population data."""
    group_desc_dict, characteristic_cols = desc_groups(pops_path)
    
    group_to_attrs = {}
    group_to_nodes = {}

    node_id = 0
    for group_id, group_info in group_desc_dict.items():
        attrs = {col: group_info[col] for col in characteristic_cols}
        group_to_attrs[group_id] = attrs
        n_nodes = int(np.ceil(scale * group_info['n']))
        group_to_nodes[group_id] = list(range(node_id, node_id + n_nodes))
        
        # Add nodes to graph
        for _ in range(n_nodes):
            G.add_node(node_id, **attrs)
            node_id += 1
    
    # Create a mapping from attrs (as a tuple of sorted items) to group_id
    attrs_to_group = {}
    for group_id, attrs in group_to_attrs.items():
        attrs_key = tuple(sorted(attrs.items()))
        attrs_to_group[attrs_key] = group_id
    
    # Store metadata in graph
    G.attrs_to_group = attrs_to_group
    G.group_to_attrs = group_to_attrs
    G.group_to_nodes = group_to_nodes
    
    # Dictionary to track links between groups
    group_ids = list(group_to_attrs.keys())
    G.existing_num_links = {(src, dst): 0 for src in group_ids for dst in group_ids}

def init_links(G, links_path, fraction, scale, reciprocity_p):
    """Initialize links in the graph based on link data."""
    
    success_bool = True
    warnings = []
    links_scale = scale**2
    
    df_n_group_links = read_file(links_path)
    print(f"Total requested links: {int(df_n_group_links['n'].sum() * links_scale)}")
   
    total_rows = len(df_n_group_links)
    for idx, row in df_n_group_links.iterrows():
        
        print(f"Row {idx + 1} of {total_rows}")
        
        # Extract source and destination attributes
        src_attrs = {k.replace('_src', ''): row[k] for k in row.index if k.endswith('_src')}
        dst_attrs = {k.replace('_dst', ''): row[k] for k in row.index if k.endswith('_dst')}
        
        num_requested_links = int(math.ceil(row['n'] * links_scale))

        # Find nodes matching the attributes
        src_nodes, src_id = find_nodes(G, **src_attrs)
        dst_nodes, dst_id = find_nodes(G, **dst_attrs)

        if not src_nodes or not dst_nodes:
            print("Group empty")
            continue 
        
        # Connect the nodes
        check_bool = connect_nodes(G, src_nodes, dst_nodes, src_id, dst_id,
                    num_requested_links, fraction, reciprocity_p)
        
        if not check_bool:
            existing_links = G.existing_num_links[(src_id, dst_id)]
            warnings.append(f"Row {idx} || Groups ({src_id})->({dst_id}) || {existing_links} >> {num_requested_links}")
    
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(warning)

def generate(pops_path, links_path, fraction, scale, reciprocity_p, use_file_based=True, base_path="graph_data"):
    """
    Generate a network using adjacency list representation.
    
    Args:
        pops_path: Path to population data
        links_path: Path to links data  
        fraction: Fraction parameter for preferential attachment
        scale: Scaling factor for population and links
        reciprocity_p: Probability of reciprocal edges
        use_file_based: If True, use file-based storage; if False, use in-memory
        base_path: Base path for file-based storage
    """
    
    print("Generating Nodes")
    
    # Choose graph implementation based on expected size
    if use_file_based:
        print("Using file-based graph storage")
        G = FileBasedGraph(base_path)
    else:
        print("Using in-memory graph storage")
        G = InMemoryGraph()
    
    init_nodes(G, pops_path, scale)
    print(f"{G.number_of_nodes()} nodes initialized")
    print()
    
    print("Generating Links")
    print("-----------------")
    init_links(G, links_path, fraction, scale, reciprocity_p)
    print("-----------------")
    print("Network Generated")
    print()
    
    # Finalize the graph
    G.finalize()
    
    return G

if __name__ == "__main__":
    pops = 'Fake_Data/fake_tab_n.xlsx'
    links = 'Data/tab_huishouden.xlsx'
    fraction = 0.4
    scale = 0.1  # Reduced for testing
    reciprocity = 1
    
    # For small scale, use in-memory; for large scale, use file-based
    use_file_based = scale > 0.1  # Use file-based for larger networks
    
    print(f"Scale: {scale}, Using file-based storage: {use_file_based}")
    
    graph = generate(pops, links, fraction, scale, reciprocity, use_file_based)
    
    # Export group information
    export_group_to_attrs_csv(graph.group_to_attrs, 'Exports/gTa_export.csv')
    export_group_to_nodes_csv(graph.group_to_nodes, "Exports/gTn_export.csv")
    
    # Save graph in Python format
    if isinstance(graph, InMemoryGraph):
        graph.save_to_file("generated_graph.pkl")
        print("Graph saved to generated_graph.pkl")
    else:
        print(f"Graph saved to {graph.base_path}/")
    
    # Export in compact non-Python formats
    print("\nExporting graph in compact formats...")
    base_name = "Exports/network"
    export_all_formats(graph, base_name)
        
    # Run statistics
    #runstats(graph)
