import numpy as np
import math
import random

# original random connections
""" def connect_nodes(G, src_nodes, dst_nodes, n_links, fraction):
    links_added = 0
    attempts = 0
    max_attempts = n_links * 10
    
    while links_added < n_links and attempts < max_attempts:
        s = random.choice(src_nodes)
        d = np.random.choice(dst_nodes)
        if s != d and not G.has_edge(s, d):
            G.add_edge(s, d)
            links_added += 1
        attempts += 1 """

# Added preferential attachment
""" def connect_nodes(G, src_nodes, dst_nodes, n_links, fraction):
    links_added = 0
    attempts = 0
    max_attempts = n_links * 10
    
    # see DB bin in thesis
    d_nodes_bin = list(np.random.choice(dst_nodes, size=(math.ceil(len(dst_nodes)*fraction)), replace=False))
        
    while links_added < n_links and attempts < max_attempts:
        s = random.choice(src_nodes)
        d_from_db = random.choice(d_nodes_bin)
        
        if s != d_from_db and not G.has_edge(s, d_from_db):
            G.add_edge(s, d_from_db)
            
            
            # NOTE
            # there seems to be discontinuity on whether this line is in the probability check or not 
            # see CLS_THESIS/Notebooks/Barabasi 2 group algorithm.ipynb VS Thesis algorithm 2
            # i'm going with this since it was present in code
            d_nodes_bin.append(random.choice(dst_nodes))
            
            if random.uniform(0,1) > fraction:
                d_nodes_bin.append(d_from_db)
                
            links_added += 1
            
        attempts += 1
         """


def connect_nodes(G, src_nodes, dst_nodes, src_id, dst_id,
                  target_link_count, fraction, reciprocity_p):
    
    success_bool = True 
    attempts = 0
    max_attempts = target_link_count * 10
    
    num_links = G.existing_num_links.get((src_id, dst_id), 0)
    
    #print(f"  Group {src_id} -> Group {dst_id}")
    #print(f"  Sizes : {len(src_nodes)}, {len(dst_nodes)}")
    #print("   Pre-existing links: ", num_links)
    #print("   Target: ", target_link_count)
    
     # TODO add handling of this, currently allows for ~20% error (plus 20 to gave more leniency for small number)
    if num_links > target_link_count + target_link_count*0.2:
        success_bool = False
        
    n0_links = num_links
    # see DB bin in thesis
    d_nodes_bin = list(np.random.choice(dst_nodes, size=(math.ceil(len(dst_nodes)*fraction)), replace=False))
        
    while num_links < target_link_count and attempts < max_attempts:
        s = random.choice(src_nodes)
        d_from_db = random.choice(d_nodes_bin)
        
        if s != d_from_db and not G.has_edge(s, d_from_db):
            G.add_edge(s, d_from_db)
            num_links += 1
            
            # reciprocity
            if random.uniform(0,1) < reciprocity_p and not G.has_edge(d_from_db, s):
                G.add_edge(d_from_db, s)
                G.existing_num_links[(dst_id, src_id)] += 1
                
            # NOTE
            # TODO check whether the prob check is correct
            # there seems to be discontinuity on whether this line is in the 1 - fraction probability check or not 
            # see CLS_THESIS/Notebooks/Barabasi 2 group algorithm.ipynb VS Thesis algorithm 2
            # i'm going with this since it was present in code
            d_nodes_bin.append(random.choice(dst_nodes))
            
            if random.uniform(0,1) > fraction:
                d_nodes_bin.append(d_from_db)
                
        attempts += 1
    print(f"   {num_links - n0_links} links added")
    G.existing_num_links[(src_id, dst_id)] += (num_links - n0_links)
    
    return success_bool