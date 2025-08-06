#!/usr/bin/env python3
"""
Simple test of export functionality.
"""

from adjacency_graph import InMemoryGraph
from graph_export import export_all_formats, export_binary_edges, export_edge_list_txt

# Create a simple test graph
print("Creating test graph...")
G = InMemoryGraph()

# Add nodes
for i in range(5):
    G.add_node(i, type="test", group=i//2)

# Add edges
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 3), (1, 4)]
for src, dst in edges:
    G.add_edge(src, dst)

print(f"Test graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Test basic export
print("\nTesting basic export...")
export_edge_list_txt(G, "test_edges.txt")
export_binary_edges(G, "test_edges.bin")

print("Export test successful!")

# Show file contents
print("\nText file contents:")
with open("test_edges.txt", "r") as f:
    print(f.read())

print("\nBinary file info:")
import os
size = os.path.getsize("test_edges.bin")
print(f"Binary file size: {size} bytes")

# Read binary file
import struct
with open("test_edges.bin", "rb") as f:
    num_nodes, num_edges = struct.unpack('II', f.read(8))
    print(f"Binary header: {num_nodes} nodes, {num_edges} edges")
    print("Binary edges:")
    for i in range(num_edges):
        src, dst = struct.unpack('II', f.read(8))
        print(f"  {src} -> {dst}")
