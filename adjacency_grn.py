import numpy as np
import math
import random

def connect_nodes(G, src_nodes, dst_nodes, src_id, dst_id,
                  target_link_count, fraction, reciprocity_p):
    """
    Connect nodes using preferential attachment algorithm.
    Optimized for adjacency list representation.
    """
    
    success_bool = True 
    attempts = 0
    max_attempts = target_link_count * 10
    
    num_links = G.existing_num_links.get((src_id, dst_id), 0)
    
    # TODO add handling of this, currently allows for ~20% error (plus 20 to give more leniency for small numbers)
    if num_links > target_link_count + target_link_count*0.2:
        success_bool = False
        
    n0_links = num_links
    # see DB bin in thesis - preferential attachment setup
    d_nodes_bin = list(np.random.choice(dst_nodes, size=(math.ceil(len(dst_nodes)*fraction)), replace=False))
        
    while num_links < target_link_count and attempts < max_attempts:
        s = random.choice(src_nodes)
        d_from_db = random.choice(d_nodes_bin)
        
        # For adjacency lists, we skip the expensive has_edge check for performance
        # Instead we allow some duplicate edges (they'll be handled by the data structure)
        if s != d_from_db:
            G.add_edge(s, d_from_db)
            num_links += 1
            
            # reciprocity - add reverse edge
            if random.uniform(0,1) < reciprocity_p:
                G.add_edge(d_from_db, s)
                G.existing_num_links[(dst_id, src_id)] += 1
                
            # Preferential attachment mechanism
            # Add random node to maintain diversity
            d_nodes_bin.append(random.choice(dst_nodes))
            
            # With probability (1-fraction), add the chosen node again (preferential attachment)
            if random.uniform(0,1) > fraction:
                d_nodes_bin.append(d_from_db)
                
        attempts += 1
    
    print(f"   {num_links - n0_links} links added")
    G.existing_num_links[(src_id, dst_id)] += (num_links - n0_links)
    
    return success_bool
