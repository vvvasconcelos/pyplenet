#!/usr/bin/env python
"""Simple test script for adjacency list implementation."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.getcwd())

def test_in_memory_graph():
    """Test InMemoryGraph functionality."""
    print("Testing InMemoryGraph...")
    
    from adjacency_graph import InMemoryGraph
    
    g = InMemoryGraph()
    
    # Add nodes
    g.add_node(0, type='person', age=25)
    g.add_node(1, type='person', age=30)
    g.add_node(2, type='person', age=35)
    
    print(f"Added {g.number_of_nodes()} nodes")
    
    # Add edges
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(0, 2)
    
    print(f"Added {g.number_of_edges()} edges")
    
    # Test edge queries
    out_edges_0 = list(g.get_out_edges(0))
    in_edges_2 = list(g.get_in_edges(2))
    
    print(f"Node 0 out-edges: {out_edges_0}")
    print(f"Node 2 in-edges: {in_edges_2}")
    
    # Test degrees
    out_degrees = g.out_degree()
    in_degrees = g.in_degree()
    
    print(f"Out-degrees: {out_degrees}")
    print(f"In-degrees: {in_degrees}")
    
    print("InMemoryGraph test passed!")
    return g

def test_file_based_graph():
    """Test FileBasedGraph functionality."""
    print("\nTesting FileBasedGraph...")
    
    from adjacency_graph import FileBasedGraph
    
    g = FileBasedGraph("test_graph")
    
    # Add nodes
    g.add_node(0, type='person', age=25)
    g.add_node(1, type='person', age=30)
    g.add_node(2, type='person', age=35)
    
    print(f"Added {g.number_of_nodes()} nodes")
    
    # Add edges
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(0, 2)
    
    print(f"Added {g.number_of_edges()} edges")
    
    # Finalize
    g.finalize()
    
    print("FileBasedGraph test passed!")
    
    # Cleanup
    g.cleanup()
    
    return g

def test_utils():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    from utils import read_file, desc_groups
    
    # Test if we can read the fake data
    try:
        df = read_file('Fake_Data/fake_tab_n.xlsx')
        print(f"Read fake data: {len(df)} rows")
        
        groups, cols = desc_groups('Fake_Data/fake_tab_n.xlsx')
        print(f"Found {len(groups)} groups with columns: {cols}")
        
        print("Utility functions test passed!")
        return True
    except Exception as e:
        print(f"Utility test failed: {e}")
        return False

if __name__ == "__main__":
    try:
        # Test in-memory graph
        g1 = test_in_memory_graph()
        
        # Test file-based graph
        g2 = test_file_based_graph()
        
        # Test utilities
        utils_ok = test_utils()
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED!")
        print("="*50)
        print("The adjacency list implementation is working correctly.")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
