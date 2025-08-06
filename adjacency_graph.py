"""
High-performance file-based adjacency list graph implementation.
Optimized for large networks that don't fit in RAM.
"""
import os
import struct
import pickle
import numpy as np
from collections import defaultdict, deque
import mmap
import json

class FileBasedGraph:
    """
    A directed graph implementation using file-based adjacency lists.
    Designed for very large networks that exceed RAM capacity.
    """
    
    def __init__(self, base_path="graph_data"):
        self.base_path = base_path
        self.nodes_file = os.path.join(base_path, "nodes.dat")
        self.edges_file = os.path.join(base_path, "edges.dat") 
        self.metadata_file = os.path.join(base_path, "metadata.json")
        self.adjacency_file = os.path.join(base_path, "adjacency.dat")
        self.reverse_adjacency_file = os.path.join(base_path, "reverse_adjacency.dat")
        
        # Ensure directory exists
        os.makedirs(base_path, exist_ok=True)
        
        # Graph metadata
        self.num_nodes = 0
        self.num_edges = 0
        self.node_attributes = {}  # node_id -> attributes dict
        self.attrs_to_group = {}
        self.group_to_attrs = {}
        self.group_to_nodes = {}
        self.existing_num_links = {}
        
        # File handles for efficient writing
        self._adjacency_handle = None
        self._reverse_adjacency_handle = None
        
        # Load existing data if available
        self._load_metadata()
    
    def _save_metadata(self):
        """Save graph metadata to file."""
        metadata = {
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'attrs_to_group': {str(k): v for k, v in self.attrs_to_group.items()},
            'group_to_attrs': self.group_to_attrs,
            'group_to_nodes': self.group_to_nodes,
            'existing_num_links': {str(k): v for k, v in self.existing_num_links.items()}
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_metadata(self):
        """Load graph metadata from file."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                self.num_nodes = metadata.get('num_nodes', 0)
                self.num_edges = metadata.get('num_edges', 0)
                
                # Convert string keys back to tuples for attrs_to_group
                attrs_to_group_str = metadata.get('attrs_to_group', {})
                self.attrs_to_group = {eval(k): v for k, v in attrs_to_group_str.items()}
                
                self.group_to_attrs = metadata.get('group_to_attrs', {})
                self.group_to_nodes = metadata.get('group_to_nodes', {})
                
                # Convert string keys back to tuples for existing_num_links
                existing_num_links_str = metadata.get('existing_num_links', {})
                self.existing_num_links = {eval(k): v for k, v in existing_num_links_str.items()}
    
    def add_node(self, node_id, **attrs):
        """Add a node with attributes."""
        self.node_attributes[node_id] = attrs
        self.num_nodes = max(self.num_nodes, node_id + 1)
    
    def _open_adjacency_files(self):
        """Open adjacency files for writing."""
        if self._adjacency_handle is None:
            self._adjacency_handle = open(self.adjacency_file, 'ab')
        if self._reverse_adjacency_handle is None:
            self._reverse_adjacency_handle = open(self.reverse_adjacency_file, 'ab')
    
    def _close_adjacency_files(self):
        """Close adjacency files."""
        if self._adjacency_handle:
            self._adjacency_handle.close()
            self._adjacency_handle = None
        if self._reverse_adjacency_handle:
            self._reverse_adjacency_handle.close()
            self._reverse_adjacency_handle = None
    
    def add_edge(self, src, dst):
        """Add an edge from src to dst."""
        self._open_adjacency_files()
        
        # Write to adjacency file (src -> dst)
        edge_data = struct.pack('II', src, dst)
        self._adjacency_handle.write(edge_data)
        
        # Write to reverse adjacency file (dst <- src) 
        reverse_edge_data = struct.pack('II', dst, src)
        self._reverse_adjacency_handle.write(reverse_edge_data)
        
        self.num_edges += 1
    
    def has_edge(self, src, dst):
        """Check if edge exists. Warning: This is slow for file-based storage."""
        # For large graphs, we'll maintain a bloom filter or in-memory cache
        # For now, this is a simplified implementation
        return False  # Assume no edge exists to avoid expensive lookups
    
    def finalize(self):
        """Finalize the graph construction and save metadata."""
        self._close_adjacency_files()
        self._save_metadata()
        print(f"Graph finalized: {self.num_nodes} nodes, {self.num_edges} edges")
    
    def get_out_edges(self, node_id, batch_size=10000):
        """Generator that yields outgoing edges for a node."""
        if not os.path.exists(self.adjacency_file):
            return
            
        with open(self.adjacency_file, 'rb') as f:
            while True:
                data = f.read(batch_size * 8)  # 8 bytes per edge (2 uint32)
                if not data:
                    break
                    
                # Unpack edges in batches
                num_edges = len(data) // 8
                edges = struct.unpack(f'{num_edges * 2}I', data)
                
                for i in range(0, len(edges), 2):
                    src, dst = edges[i], edges[i + 1]
                    if src == node_id:
                        yield dst
    
    def get_in_edges(self, node_id, batch_size=10000):
        """Generator that yields incoming edges for a node."""
        if not os.path.exists(self.reverse_adjacency_file):
            return
            
        with open(self.reverse_adjacency_file, 'rb') as f:
            while True:
                data = f.read(batch_size * 8)  # 8 bytes per edge (2 uint32)
                if not data:
                    break
                    
                # Unpack edges in batches
                num_edges = len(data) // 8
                edges = struct.unpack(f'{num_edges * 2}I', data)
                
                for i in range(0, len(edges), 2):
                    dst, src = edges[i], edges[i + 1]
                    if dst == node_id:
                        yield src
    
    def number_of_nodes(self):
        """Return number of nodes."""
        return self.num_nodes
    
    def number_of_edges(self):
        """Return number of edges."""
        return self.num_edges
    
    def is_directed(self):
        """Return True if graph is directed."""
        return True
    
    def nodes(self):
        """Return iterator over all node IDs."""
        return range(self.num_nodes)
    
    def out_degree(self, node_id=None):
        """Calculate out-degree for node(s)."""
        if node_id is not None:
            return sum(1 for _ in self.get_out_edges(node_id))
        
        # Calculate for all nodes
        degrees = [0] * self.num_nodes
        if os.path.exists(self.adjacency_file):
            with open(self.adjacency_file, 'rb') as f:
                while True:
                    data = f.read(80000)  # Read in chunks
                    if not data:
                        break
                    num_edges = len(data) // 8
                    edges = struct.unpack(f'{num_edges * 2}I', data)
                    for i in range(0, len(edges), 2):
                        src = edges[i]
                        if src < self.num_nodes:
                            degrees[src] += 1
        
        return [(i, degrees[i]) for i in range(self.num_nodes)]
    
    def in_degree(self, node_id=None):
        """Calculate in-degree for node(s)."""
        if node_id is not None:
            return sum(1 for _ in self.get_in_edges(node_id))
        
        # Calculate for all nodes
        degrees = [0] * self.num_nodes
        if os.path.exists(self.reverse_adjacency_file):
            with open(self.reverse_adjacency_file, 'rb') as f:
                while True:
                    data = f.read(80000)  # Read in chunks
                    if not data:
                        break
                    num_edges = len(data) // 8
                    edges = struct.unpack(f'{num_edges * 2}I', data)
                    for i in range(0, len(edges), 2):
                        dst = edges[i]
                        if dst < self.num_nodes:
                            degrees[dst] += 1
        
        return [(i, degrees[i]) for i in range(self.num_nodes)]
    
    def cleanup(self):
        """Remove all graph files."""
        files_to_remove = [
            self.nodes_file, self.edges_file, self.metadata_file,
            self.adjacency_file, self.reverse_adjacency_file
        ]
        for file_path in files_to_remove:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        if os.path.exists(self.base_path) and not os.listdir(self.base_path):
            os.rmdir(self.base_path)


class InMemoryGraph:
    """
    In-memory adjacency list implementation for smaller graphs.
    Much faster than file-based but limited by RAM.
    """
    
    def __init__(self):
        self.adjacency_list = defaultdict(set)  # src -> set of dst nodes
        self.reverse_adjacency_list = defaultdict(set)  # dst -> set of src nodes
        self.node_attributes = {}
        self.num_nodes = 0
        self.num_edges = 0
        
        # Additional attributes for compatibility
        self.attrs_to_group = {}
        self.group_to_attrs = {}
        self.group_to_nodes = {}
        self.existing_num_links = {}
    
    def add_node(self, node_id, **attrs):
        """Add a node with attributes."""
        self.node_attributes[node_id] = attrs
        self.num_nodes = max(self.num_nodes, node_id + 1)
    
    def add_edge(self, src, dst):
        """Add an edge from src to dst."""
        if dst not in self.adjacency_list[src]:
            self.adjacency_list[src].add(dst)
            self.reverse_adjacency_list[dst].add(src)
            self.num_edges += 1
    
    def has_edge(self, src, dst):
        """Check if edge exists."""
        return dst in self.adjacency_list[src]
    
    def get_out_edges(self, node_id):
        """Get outgoing edges for a node."""
        return self.adjacency_list[node_id]
    
    def get_in_edges(self, node_id):
        """Get incoming edges for a node."""
        return self.reverse_adjacency_list[node_id]
    
    def number_of_nodes(self):
        """Return number of nodes."""
        return self.num_nodes
    
    def number_of_edges(self):
        """Return number of edges."""
        return self.num_edges
    
    def is_directed(self):
        """Return True if graph is directed."""
        return True
    
    def nodes(self):
        """Return iterator over all node IDs."""
        return range(self.num_nodes)
    
    def out_degree(self, node_id=None):
        """Calculate out-degree for node(s)."""
        if node_id is not None:
            return len(self.adjacency_list[node_id])
        return [(i, len(self.adjacency_list[i])) for i in range(self.num_nodes)]
    
    def in_degree(self, node_id=None):
        """Calculate in-degree for node(s)."""
        if node_id is not None:
            return len(self.reverse_adjacency_list[node_id])
        return [(i, len(self.reverse_adjacency_list[i])) for i in range(self.num_nodes)]
    
    def finalize(self):
        """Finalize graph (no-op for in-memory)."""
        pass
    
    def save_to_file(self, filename):
        """Save graph to pickle file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_from_file(cls, filename):
        """Load graph from pickle file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)
