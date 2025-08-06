from utils import find_nodes, read_file, desc_groups, export_group_to_attrs_csv, export_group_to_nodes_csv
from GRN import connect_nodes
from netstats import runstats
import numpy as np
import networkx as nx
import math

# Take in file tab_buren.xlsx 
# Generate network using the mixed 2 group network thing from kamiel

# Create a dict that stores the characteristics of each group
# Create a dict that stores the group each node belongs to 

def init_nodes(G, pops_path, scale):
    group_desc_dict, characteristic_cols = desc_groups(pops_path)
    
    group_to_attrs = {}
    group_to_nodes = {}

    node_id = 0
    for group_id, group_info in group_desc_dict.items():
        attrs = {col: group_info[col] for col in characteristic_cols}
        group_to_attrs[group_id] = attrs
        n_nodes = int(np.ceil(scale * group_info['n']))
        group_to_nodes[group_id] = list(range(node_id, node_id + n_nodes))
        for _ in range(n_nodes):
            G.add_node(node_id, **attrs)
            node_id += 1
    
    # Create a mapping from attrs (as a tuple of sorted items) to group_id
    attrs_to_group = {}
    for group_id, attrs in group_to_attrs.items():
        attrs_key = tuple(sorted(attrs.items()))
        attrs_to_group[attrs_key] = group_id
    
    G.attrs_to_group = attrs_to_group
    G.group_to_attrs = group_to_attrs
    G.group_to_nodes = group_to_nodes
    
    # a dict of tuple (group_id, group_id) : num storing the existing number of links between two groups.
    # this is used to record the number of links generated.
    # useful cause when reciprocity happens we want to count that link too.
    group_ids = list(group_to_attrs.keys())
    G.existing_num_links = {(src, dst): 0 for src in group_ids for dst in group_ids}
    
    

def init_links(G, links_path, fraction, scale, reciprocity_p):
    
    success_bool = True
    warnings = []
    links_scale = scale**2
    
    df_n_group_links = read_file(links_path)
    print(f"Total requested links: {int(df_n_group_links['n'].sum() * links_scale)}")
   
    total_rows = len(df_n_group_links)
    for idx, row in df_n_group_links.iterrows():
        
        print(f"Row {idx + 1} of {total_rows}")
        
        src_attrs = {k.replace('_src', ''): row[k] for k in row.index if k.endswith('_src')}
        dst_attrs = {k.replace('_dst', ''): row[k] for k in row.index if k.endswith('_dst')}
        
        num_requested_links = int(math.ceil(row['n'] * links_scale))

        src_nodes, src_id = find_nodes(G, **src_attrs)
        dst_nodes, dst_id = find_nodes(G, **dst_attrs)

        if not src_nodes or not dst_nodes:
            print("Group empty")
            continue 
        
        
        check_bool = connect_nodes(G, src_nodes, dst_nodes, src_id, dst_id,
                    num_requested_links, fraction, reciprocity_p)
        
        if not check_bool:
            existing_links = G.existing_num_links[(src_id, dst_id)]
            warnings.append(f"Row {idx} || Groups ({src_id})->({dst_id}) || {existing_links} >> {num_requested_links}")
    
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(warning)
    

def generate(pops_path, links_path, fraction, scale, reciprocity_p):
    
    print("Generating Nodes")
    G = nx.DiGraph()
    init_nodes(G, pops_path, scale)
    print(f"{G.number_of_nodes()} nodes initialized")
    print()
    print("Generating Links")
    print("-----------------")
    init_links(G, links_path, fraction, scale, reciprocity_p)
    print("-----------------")
    print("Network Generated")
    print()
    return G
    
    
if __name__ == "__main__":
    pops = 'Fake_Data/fake_tab_n.xlsx'
    links = 'Data/tab_huishouden.xlsx'
    #pops = 'testdata_n.csv'
    #links = 'testdata_links.csv'
    fraction = 0.4
    scale = 0.1
    reciprocity = 1
    graph = generate(pops, links, fraction, scale, reciprocity)
    
    export_group_to_attrs_csv(graph.group_to_attrs, 'gTa_export.csv')
    export_group_to_nodes_csv(graph.group_to_nodes, "gTn_export.csv")
    import pickle

    with open("generated_graph.gpickle", "wb") as f:
        pickle.dump(graph, f)
        
    runstats(graph)