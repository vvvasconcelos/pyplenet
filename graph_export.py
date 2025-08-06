"""
Export functions for adjacency list graphs to compact, non-Python formats.
Supports multiple standard graph file formats.
"""
import struct
import gzip
import csv

def export_edge_list_txt(G, filename, compressed=False):
    """
    Export graph as simple edge list text file.
    Format: each line contains "src dst"
    
    Args:
        G: Graph object (InMemoryGraph or FileBasedGraph)
        filename: Output filename
        compressed: If True, compress with gzip
    """
    open_func = gzip.open if compressed else open
    mode = 'wt' if compressed else 'w'
    
    with open_func(filename, mode) as f:
        # Write header comment
        f.write(f"# Directed graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")
        f.write("# Format: src dst\n")
        
        if hasattr(G, 'adjacency_file'):
            # File-based graph
            import os
            if os.path.exists(G.adjacency_file):
                with open(G.adjacency_file, 'rb') as adj_file:
                    while True:
                        data = adj_file.read(80000)  # Read in chunks
                        if not data:
                            break
                        num_edges = len(data) // 8
                        edges = struct.unpack(f'{num_edges * 2}I', data)
                        for i in range(0, len(edges), 2):
                            src, dst = edges[i], edges[i + 1]
                            f.write(f"{src} {dst}\n")
        else:
            # In-memory graph
            for src in range(G.number_of_nodes()):
                for dst in G.get_out_edges(src):
                    f.write(f"{src} {dst}\n")
    
    print(f"Edge list exported to {filename}")

def export_adjacency_list_txt(G, filename, compressed=False):
    """
    Export graph as adjacency list text file.
    Format: each line contains "src: dst1 dst2 dst3 ..."
    
    Args:
        G: Graph object
        filename: Output filename  
        compressed: If True, compress with gzip
    """
    open_func = gzip.open if compressed else open
    mode = 'wt' if compressed else 'w'
    
    with open_func(filename, mode) as f:
        # Write header comment
        f.write(f"# Directed adjacency list: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")
        f.write("# Format: src: dst1 dst2 dst3 ...\n")
        
        if hasattr(G, 'adjacency_file'):
            # File-based graph - read adjacency file directly and group by source
            import os
            from collections import defaultdict
            
            if os.path.exists(G.adjacency_file):
                # Build adjacency lists by reading file once
                adjacency_lists = defaultdict(list)
                
                with open(G.adjacency_file, 'rb') as adj_file:
                    while True:
                        data = adj_file.read(80000)  # Read in chunks
                        if not data:
                            break
                        num_edges = len(data) // 8
                        edges = struct.unpack(f'{num_edges * 2}I', data)
                        for i in range(0, len(edges), 2):
                            src, dst = edges[i], edges[i + 1]
                            adjacency_lists[src].append(dst)
                
                # Write adjacency lists for all nodes
                for src in range(G.number_of_nodes()):
                    neighbors = adjacency_lists.get(src, [])
                    if neighbors:
                        neighbors_str = " ".join(map(str, neighbors))
                        f.write(f"{src}: {neighbors_str}\n")
                    else:
                        f.write(f"{src}:\n")  # Isolated node
            else:
                # No adjacency file, write empty lists
                for src in range(G.number_of_nodes()):
                    f.write(f"{src}:\n")
        else:
            # In-memory graph - use existing adjacency lists
            for src in range(G.number_of_nodes()):
                neighbors = list(G.get_out_edges(src))
                if neighbors:
                    neighbors_str = " ".join(map(str, neighbors))
                    f.write(f"{src}: {neighbors_str}\n")
                else:
                    f.write(f"{src}:\n")  # Isolated node
    
    print(f"Adjacency list exported to {filename}")

def export_binary_edges(G, filename):
    """
    Export graph as binary edge list.
    Format: binary file with 8 bytes per edge (src:4bytes, dst:4bytes)
    Very compact and fast to read.
    
    Args:
        G: Graph object
        filename: Output filename
    """
    with open(filename, 'wb') as f:
        # Write header: num_nodes (4 bytes), num_edges (4 bytes)
        f.write(struct.pack('II', G.number_of_nodes(), G.number_of_edges()))
        
        if hasattr(G, 'adjacency_file'):
            # File-based graph - copy binary data directly
            import os
            if os.path.exists(G.adjacency_file):
                with open(G.adjacency_file, 'rb') as adj_file:
                    while True:
                        chunk = adj_file.read(65536)  # 64KB chunks
                        if not chunk:
                            break
                        f.write(chunk)
        else:
            # In-memory graph
            for src in range(G.number_of_nodes()):
                for dst in G.get_out_edges(src):
                    f.write(struct.pack('II', src, dst))
    
    print(f"Binary edge list exported to {filename}")

def export_csv_edges(G, filename, include_attributes=False):
    """
    Export graph as CSV edge list.
    
    Args:
        G: Graph object
        filename: Output filename
        include_attributes: If True, include node attributes
    """
    with open(filename, 'w', newline='') as f:
        if include_attributes:
            writer = csv.writer(f)
            # Write header
            if G.node_attributes:
                sample_attrs = next(iter(G.node_attributes.values()))
                attr_names = list(sample_attrs.keys())
                writer.writerow(['src', 'dst'] + [f'src_{attr}' for attr in attr_names] + [f'dst_{attr}' for attr in attr_names])
            else:
                writer.writerow(['src', 'dst'])
            
            # Write edges with attributes
            if hasattr(G, 'adjacency_file'):
                # File-based graph
                import os
                if os.path.exists(G.adjacency_file):
                    with open(G.adjacency_file, 'rb') as adj_file:
                        while True:
                            data = adj_file.read(80000)
                            if not data:
                                break
                            num_edges = len(data) // 8
                            edges = struct.unpack(f'{num_edges * 2}I', data)
                            for i in range(0, len(edges), 2):
                                src, dst = edges[i], edges[i + 1]
                                row = [src, dst]
                                if G.node_attributes:
                                    src_attrs = G.node_attributes.get(src, {})
                                    dst_attrs = G.node_attributes.get(dst, {})
                                    for attr in attr_names:
                                        row.append(src_attrs.get(attr, ''))
                                        row.append(dst_attrs.get(attr, ''))
                                writer.writerow(row)
            else:
                # In-memory graph
                for src in range(G.number_of_nodes()):
                    for dst in G.get_out_edges(src):
                        row = [src, dst]
                        if G.node_attributes:
                            src_attrs = G.node_attributes.get(src, {})
                            dst_attrs = G.node_attributes.get(dst, {})
                            for attr in attr_names:
                                row.append(src_attrs.get(attr, ''))
                                row.append(dst_attrs.get(attr, ''))
                        writer.writerow(row)
        else:
            # Simple edge list CSV
            writer = csv.writer(f)
            writer.writerow(['src', 'dst'])
            
            if hasattr(G, 'adjacency_file'):
                # File-based graph
                import os
                if os.path.exists(G.adjacency_file):
                    with open(G.adjacency_file, 'rb') as adj_file:
                        while True:
                            data = adj_file.read(80000)
                            if not data:
                                break
                            num_edges = len(data) // 8
                            edges = struct.unpack(f'{num_edges * 2}I', data)
                            for i in range(0, len(edges), 2):
                                src, dst = edges[i], edges[i + 1]
                                writer.writerow([src, dst])
            else:
                # In-memory graph
                for src in range(G.number_of_nodes()):
                    for dst in G.get_out_edges(src):
                        writer.writerow([src, dst])
    
    print(f"CSV edge list exported to {filename}")

def export_mtx_format(G, filename):
    """
    Export graph in Matrix Market (.mtx) format.
    Standard sparse matrix format, widely supported.
    
    Args:
        G: Graph object
        filename: Output filename
    """
    with open(filename, 'w') as f:
        # Write header
        f.write("%%MatrixMarket matrix coordinate integer general\n")
        f.write(f"{G.number_of_nodes()} {G.number_of_nodes()} {G.number_of_edges()}\n")
        
        if hasattr(G, 'adjacency_file'):
            # File-based graph
            import os
            if os.path.exists(G.adjacency_file):
                with open(G.adjacency_file, 'rb') as adj_file:
                    while True:
                        data = adj_file.read(80000)
                        if not data:
                            break
                        num_edges = len(data) // 8
                        edges = struct.unpack(f'{num_edges * 2}I', data)
                        for i in range(0, len(edges), 2):
                            src, dst = edges[i], edges[i + 1]
                            # MTX format is 1-indexed
                            f.write(f"{src + 1} {dst + 1} 1\n")
        else:
            # In-memory graph
            for src in range(G.number_of_nodes()):
                for dst in G.get_out_edges(src):
                    # MTX format is 1-indexed
                    f.write(f"{src + 1} {dst + 1} 1\n")
    
    print(f"Matrix Market format exported to {filename}")

def export_snap_format(G, filename):
    """
    Export graph in SNAP format (Stanford Network Analysis Platform).
    Simple format: # comments followed by edge list.
    
    Args:
        G: Graph object
        filename: Output filename
    """
    with open(filename, 'w') as f:
        # Write SNAP header
        f.write(f"# Directed graph (each unordered pair of nodes is saved once): {filename}\n")
        f.write(f"# Nodes: {G.number_of_nodes()} Edges: {G.number_of_edges()}\n")
        f.write(f"# FromNodeId	ToNodeId\n")
        
        if hasattr(G, 'adjacency_file'):
            # File-based graph
            import os
            if os.path.exists(G.adjacency_file):
                with open(G.adjacency_file, 'rb') as adj_file:
                    while True:
                        data = adj_file.read(80000)
                        if not data:
                            break
                        num_edges = len(data) // 8
                        edges = struct.unpack(f'{num_edges * 2}I', data)
                        for i in range(0, len(edges), 2):
                            src, dst = edges[i], edges[i + 1]
                            f.write(f"{src}\t{dst}\n")
        else:
            # In-memory graph
            for src in range(G.number_of_nodes()):
                for dst in G.get_out_edges(src):
                    f.write(f"{src}\t{dst}\n")
    
    print(f"SNAP format exported to {filename}")

def export_all_formats(G, base_filename):
    """
    Export graph in all supported formats for convenience.
    
    Args:
        G: Graph object
        base_filename: Base name for output files (extensions will be added)
    """
    print(f"Exporting graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges) in multiple formats...")
    
    # Text formats
    export_edge_list_txt(G, f"{base_filename}_edges.txt")
    export_edge_list_txt(G, f"{base_filename}_edges.txt.gz", compressed=True)
    export_adjacency_list_txt(G, f"{base_filename}_adj.txt")
    
    # CSV format
    export_csv_edges(G, f"{base_filename}_edges.csv")
    
    # Binary format (most compact)
    export_binary_edges(G, f"{base_filename}_edges.bin")
    
    # Standard formats
    export_mtx_format(G, f"{base_filename}.mtx")
    export_snap_format(G, f"{base_filename}.snap")
    
    print("All formats exported successfully!")
    
    # Print file sizes for comparison
    import os
    formats = [
        (f"{base_filename}_edges.txt", "Edge list (text)"),
        (f"{base_filename}_edges.txt.gz", "Edge list (compressed)"),
        (f"{base_filename}_adj.txt", "Adjacency list (text)"),
        (f"{base_filename}_edges.csv", "Edge list (CSV)"),
        (f"{base_filename}_edges.bin", "Binary edge list"),
        (f"{base_filename}.mtx", "Matrix Market"),
        (f"{base_filename}.snap", "SNAP format")
    ]
    
    print("\nFile size comparison:")
    for filename, description in formats:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            if size > 1024*1024:
                size_str = f"{size/(1024*1024):.1f} MB"
            elif size > 1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size} bytes"
            print(f"  {description}: {size_str}")

# Reading functions for the exported formats
def read_binary_edges(filename):
    """
    Read binary edge list file.
    Returns: (num_nodes, num_edges, edge_iterator)
    """
    with open(filename, 'rb') as f:
        # Read header
        header = f.read(8)
        num_nodes, num_edges = struct.unpack('II', header)
        
        def edge_iterator():
            while True:
                data = f.read(8)
                if not data:
                    break
                src, dst = struct.unpack('II', data)
                yield src, dst
        
        return num_nodes, num_edges, edge_iterator()

def read_edge_list_txt(filename):
    """
    Read text edge list file.
    Returns: edge_iterator
    """
    def edge_iterator():
        open_func = gzip.open if filename.endswith('.gz') else open
        mode = 'rt' if filename.endswith('.gz') else 'r'
        
        with open_func(filename, mode) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 2:
                        yield int(parts[0]), int(parts[1])
    
    return edge_iterator()
