#!/usr/bin/env python3
"""
Utility script to read and inspect exported graph formats.
"""

import os
import struct
import gzip

def inspect_binary_format(filename):
    """Inspect binary graph file."""
    print(f"\n=== Binary Format: {filename} ===")
    if not os.path.exists(filename):
        print("File not found!")
        return
    
    with open(filename, 'rb') as f:
        # Read header
        header = f.read(8)
        if len(header) < 8:
            print("Invalid binary file!")
            return
        
        num_nodes, num_edges = struct.unpack('II', header)
        print(f"Nodes: {num_nodes}")
        print(f"Edges: {num_edges}")
        print(f"File size: {os.path.getsize(filename)} bytes")
        print(f"Expected size: {num_edges * 8 + 8} bytes")
        
        # Read first few edges
        print("First 10 edges:")
        for i in range(min(10, num_edges)):
            edge_data = f.read(8)
            if len(edge_data) < 8:
                break
            src, dst = struct.unpack('II', edge_data)
            print(f"  {src} -> {dst}")

def inspect_text_format(filename):
    """Inspect text graph file."""
    print(f"\n=== Text Format: {filename} ===")
    if not os.path.exists(filename):
        print("File not found!")
        return
    
    open_func = gzip.open if filename.endswith('.gz') else open
    mode = 'rt' if filename.endswith('.gz') else 'r'
    
    edge_count = 0
    with open_func(filename, mode) as f:
        print("First 10 lines:")
        for i, line in enumerate(f):
            if i < 10:
                print(f"  {line.rstrip()}")
            if not line.startswith('#') and line.strip():
                edge_count += 1
            if i >= 20:  # Don't read entire file for large graphs
                break
    
    print(f"File size: {os.path.getsize(filename)} bytes")
    print(f"Estimated edges: {edge_count} (from first 20 lines)")

def compare_all_formats(base_name="network"):
    """Compare all exported graph formats."""
    formats = [
        (f"{base_name}_edges.txt", "Edge list (text)", inspect_text_format),
        (f"{base_name}_edges.txt.gz", "Edge list (compressed)", inspect_text_format),
        (f"{base_name}_adj.txt", "Adjacency list (text)", inspect_text_format),
        (f"{base_name}_edges.csv", "Edge list (CSV)", inspect_text_format),
        (f"{base_name}_edges.bin", "Binary edge list", inspect_binary_format),
        (f"{base_name}.mtx", "Matrix Market", inspect_text_format),
        (f"{base_name}.snap", "SNAP format", inspect_text_format),
        ("network_compact.bin", "Compact binary", inspect_binary_format)
    ]
    
    print("=== GRAPH FORMAT COMPARISON ===")
    
    # File size comparison
    print("\nFile sizes:")
    existing_files = []
    for filename, description, _ in formats:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            if size > 1024*1024:
                size_str = f"{size/(1024*1024):.1f} MB"
            elif size > 1024:
                size_str = f"{size/1024:.1f} KB"  
            else:
                size_str = f"{size} bytes"
            print(f"  {description:20s}: {size_str:>10s}")
            existing_files.append((filename, description, size))
    
    # Find most compact
    if existing_files:
        smallest = min(existing_files, key=lambda x: x[2])
        print(f"\nMost compact: {smallest[1]} ({smallest[2]} bytes)")
    
    # Detailed inspection of each format
    for filename, description, inspect_func in formats:
        if os.path.exists(filename):
            inspect_func(filename)

def read_binary_graph_simple(filename):
    """Simple function to read binary graph format."""
    print(f"\n=== Reading Binary Graph: {filename} ===")
    
    edges = []
    with open(filename, 'rb') as f:
        # Read header
        num_nodes, num_edges = struct.unpack('II', f.read(8))
        print(f"Graph: {num_nodes} nodes, {num_edges} edges")
        
        # Read all edges
        for _ in range(num_edges):
            edge_data = f.read(8)
            if len(edge_data) < 8:
                break
            src, dst = struct.unpack('II', edge_data)
            edges.append((src, dst))
    
    print(f"Successfully read {len(edges)} edges")
    return num_nodes, edges

def convert_binary_to_text(binary_file, text_file):
    """Convert binary format to simple text format."""
    print(f"Converting {binary_file} to {text_file}...")
    
    num_nodes, edges = read_binary_graph_simple(binary_file)
    
    with open(text_file, 'w') as f:
        f.write(f"# Graph: {num_nodes} nodes, {len(edges)} edges\n")
        f.write("# Format: src dst\n")
        for src, dst in edges:
            f.write(f"{src} {dst}\n")
    
    print(f"Converted successfully! Text file: {text_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "compare":
            compare_all_formats()
        elif command == "binary" and len(sys.argv) > 2:
            inspect_binary_format(sys.argv[2])
        elif command == "text" and len(sys.argv) > 2:
            inspect_text_format(sys.argv[2])
        elif command == "convert" and len(sys.argv) > 3:
            convert_binary_to_text(sys.argv[2], sys.argv[3])
        elif command == "read" and len(sys.argv) > 2:
            read_binary_graph_simple(sys.argv[2])
        else:
            print("Usage:")
            print("  python graph_inspect.py compare")
            print("  python graph_inspect.py binary <file.bin>")
            print("  python graph_inspect.py text <file.txt>")
            print("  python graph_inspect.py convert <input.bin> <output.txt>")
            print("  python graph_inspect.py read <file.bin>")
    else:
        # Default: compare all formats if they exist
        compare_all_formats()
