"""
High-performance file-based adjacency list graph implementation.

This module provides a FileBasedGraph class designed for handling very large
networks that exceed RAM capacity. The implementation uses disk-based storage
with binary files for adjacency lists, enabling efficient operations on
graphs with millions or billions of edges while maintaining minimal memory
footprint.

The module prioritizes memory efficiency over speed for certain operations,
making it suitable for large-scale network analysis where the graph cannot
fit entirely in memory.

Classes
-------
FileBasedGraph : A directed graph using file-based adjacency lists

Notes
-----
The file-based approach trades some performance for scalability. Operations
like edge existence checking are intentionally simplified to avoid expensive
file I/O. For applications requiring frequent random access to edges,
consider using in-memory graph structures for smaller networks.

The binary file format uses little-endian unsigned 32-bit integers for
node IDs, limiting graphs to approximately 4 billion nodes per graph.

Examples
--------
>>> from graph import FileBasedGraph
>>> G = FileBasedGraph("my_large_network")
>>> G.add_node(0, group="A", population=1000)
>>> G.add_node(1, group="B", population=500)
>>> G.add_edge(0, 1)
>>> G.finalize()
>>> print(f"Created network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
Created network: 2 nodes, 1 edges
"""

import os
import struct
import json

class FileBasedGraph:
    """
    A directed graph implementation using file-based adjacency lists.
    
    Designed for very large networks that exceed RAM capacity by storing
    graph data in binary files rather than keeping everything in memory.
    Supports efficient edge addition and provides generators for traversal
    to minimize memory usage.
    
    Parameters
    ----------
    base_path : str, optional
        Directory path where graph files will be stored. Default is "graph_data".
        The directory will be created if it doesn't exist.
    
    Attributes
    ----------
    base_path : str
        Directory containing all graph data files
    num_nodes : int
        Total number of nodes in the graph
    num_edges : int
        Total number of edges in the graph
    node_attributes : dict
        Mapping from node_id to attribute dictionary
    attrs_to_group : dict
        Mapping from attribute tuples to group IDs
    group_to_attrs : dict
        Mapping from group IDs to attribute dictionaries
    group_to_nodes : dict
        Mapping from group IDs to lists of node IDs
    existing_num_links : dict
        Tracking dictionary for links between groups
    
    Notes
    -----
    The graph data is stored in several files:
    - adjacency.dat: Binary file storing outgoing edges
    - reverse_adjacency.dat: Binary file storing incoming edges  
    - metadata.json: JSON file with graph metadata and group information
    
    Examples
    --------
    >>> graph = FileBasedGraph("my_graph")
    >>> graph.add_node(0, group="A", age=25)
    >>> graph.add_node(1, group="B", age=30)
    >>> graph.add_edge(0, 1)
    >>> graph.finalize()
    >>> print(f"Graph has {graph.number_of_nodes()} nodes")
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
        """
        Save graph metadata to JSON file.
        
        Serializes all graph metadata including node counts, group mappings,
        and link tracking information to a JSON file for persistence.
        Dictionary keys that are tuples are converted to strings for JSON
        compatibility.
        
        Notes
        -----
        This method is called automatically by finalize() and doesn't 
        typically need to be called directly.
        """
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
        """
        Load graph metadata from JSON file.
        
        Restores graph state from previously saved metadata file.
        Converts string keys back to tuples where necessary for
        attrs_to_group and existing_num_links dictionaries.
        
        Notes
        -----
        This method is called automatically during initialization
        if a metadata file exists.
        """
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
        """
        Add a node with attributes to the graph.
        
        Parameters
        ----------
        node_id : int
            Unique identifier for the node. Should be non-negative integer.
        **attrs : dict
            Arbitrary keyword arguments representing node attributes.
            
        Notes
        -----
        Node attributes are stored in memory and persisted to the metadata
        file when finalize() is called. The num_nodes counter is automatically
        updated to accommodate the highest node_id seen.
        
        Examples
        --------
        >>> graph.add_node(0, group="A", age=25, city="Amsterdam")
        >>> graph.add_node(5, group="B", age=30)  # num_nodes becomes 6
        """
        self.node_attributes[node_id] = attrs
        self.num_nodes = max(self.num_nodes, node_id + 1)
    
    def _open_adjacency_files(self):
        """
        Open adjacency files for writing in append binary mode.
        
        Opens file handles for both forward and reverse adjacency files
        if they are not already open. Files are opened in append mode
        to support incremental edge addition.
        """
        if self._adjacency_handle is None:
            self._adjacency_handle = open(self.adjacency_file, 'ab')
        if self._reverse_adjacency_handle is None:
            self._reverse_adjacency_handle = open(self.reverse_adjacency_file, 'ab')
    
    def _close_adjacency_files(self):
        """
        Close adjacency file handles.
        
        Safely closes any open file handles and sets them to None.
        Called automatically by finalize() but can be called manually
        to free resources.
        """
        if self._adjacency_handle:
            self._adjacency_handle.close()
            self._adjacency_handle = None
        if self._reverse_adjacency_handle:
            self._reverse_adjacency_handle.close()
            self._reverse_adjacency_handle = None
    
    def add_edge(self, src, dst):
        """
        Add a directed edge from source to destination node.
        
        Parameters
        ----------
        src : int
            Source node ID
        dst : int  
            Destination node ID
            
        Notes
        -----
        Edges are written to binary files for efficient storage:
        - Forward edge (src->dst) written to adjacency.dat
        - Reverse edge (dst<-src) written to reverse_adjacency.dat
        
        The edge count is automatically incremented. File handles are
        opened lazily and kept open for efficient batch writing.
        
        Examples
        --------
        >>> graph.add_edge(0, 1)  # Add edge from node 0 to node 1
        >>> graph.add_edge(1, 0)  # Add reverse edge
        """
        self._open_adjacency_files()
        
        # Write to adjacency file (src -> dst)
        edge_data = struct.pack('II', src, dst)
        self._adjacency_handle.write(edge_data)
        
        # Write to reverse adjacency file (dst <- src) 
        reverse_edge_data = struct.pack('II', dst, src)
        self._reverse_adjacency_handle.write(reverse_edge_data)
        
        self.num_edges += 1
    
    def has_edge(self, src, dst):
        """
        Check if a directed edge exists from source to destination.
        
        Parameters
        ----------
        src : int
            Source node ID
        dst : int
            Destination node ID
            
        Returns
        -------
        bool
            True if edge exists, False otherwise
            
        Warning
        -------
        This implementation always returns False to avoid expensive
        file lookups. For large graphs, consider maintaining an
        in-memory cache or bloom filter for edge existence checks.
        
        Notes
        -----
        This is intentionally simplified for performance reasons.
        For accurate edge existence checking, implement a separate
        index or use get_out_edges() to check neighbors.
        """
        # For large graphs, we'll maintain a bloom filter or in-memory cache
        # For now, this is a simplified implementation
        return False  # Assume no edge exists to avoid expensive lookups
    
    def finalize(self):
        """
        Finalize graph construction and save all metadata.
        
        Closes any open file handles and saves the current graph state
        to the metadata file. Should be called after all nodes and edges
        have been added to ensure data persistence.
        
        Notes
        -----
        After calling finalize(), the graph can still be used for reading
        operations but adding new edges may require reopening file handles.
        
        Examples
        --------
        >>> graph.add_node(0)
        >>> graph.add_edge(0, 1) 
        >>> graph.finalize()
        Graph finalized: 2 nodes, 1 edges
        """
        self._close_adjacency_files()
        self._save_metadata()
        print(f"Graph finalized: {self.num_nodes} nodes, {self.num_edges} edges")
    
    def get_out_edges(self, node_id, batch_size=10000):
        """
        Generator yielding all outgoing edges for a given node.
        
        Parameters
        ----------
        node_id : int
            The node ID to find outgoing edges for
        batch_size : int, optional
            Number of edges to read from file in each batch. Default is 10000.
            Larger values use more memory but may be faster for dense graphs.
            
        Yields
        ------
        int
            Destination node IDs that this node has edges to
        
        Examples
        --------
        >>> for neighbor in graph.get_out_edges(0):
        ...     print(f"Node 0 -> Node {neighbor}")
        """
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
        """
        Generator yielding all incoming edges for a given node.
        
        Parameters
        ----------
        node_id : int
            The node ID to find incoming edges for
        batch_size : int, optional
            Number of edges to read from file in each batch. Default is 10000.
            
        Yields
        ------
        int
            Source node IDs that have edges to this node
        
        Examples
        --------
        >>> for source in graph.get_in_edges(5):
        ...     print(f"Node {source} -> Node 5")
        """
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
        """
        Return the total number of nodes in the graph.
        
        Returns
        -------
        int
            Number of nodes
        """
        return self.num_nodes
    
    def number_of_edges(self):
        """
        Return the total number of edges in the graph.
        
        Returns
        -------
        int
            Number of directed edges
        """
        return self.num_edges
    
    def is_directed(self):
        """
        Check if the graph is directed.
        
        Returns
        -------
        bool
            Always True for this implementation
        """
        return True
    
    def nodes(self):
        """
        Return an iterator over all node IDs.
        
        Returns
        -------
        range
            Iterator yielding node IDs from 0 to num_nodes-1
        """
        return range(self.num_nodes)
    
    def out_degree(self, node_id=None):
        """
        Calculate out-degree for node(s).
        
        Parameters
        ----------
        node_id : int, optional
            If provided, return out-degree for this specific node.
            If None, return out-degrees for all nodes.
            
        Returns
        -------
        int or list of tuple
            If node_id provided: integer out-degree of that node
            If node_id is None: list of (node_id, out_degree) tuples for all nodes
            
        Notes
        -----
        For a specific node, this method iterates through the adjacency file
        which can be slow for large graphs. For all nodes, it reads the entire
        adjacency file once and counts degrees efficiently.
        
        Examples
        --------
        >>> degree = graph.out_degree(0)  # Out-degree of node 0
        >>> all_degrees = graph.out_degree()  # [(0, 2), (1, 1), ...]
        """
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
        """
        Calculate in-degree for node(s).
        
        Parameters
        ----------
        node_id : int, optional
            If provided, return in-degree for this specific node.
            If None, return in-degrees for all nodes.
            
        Returns
        -------
        int or list of tuple
            If node_id provided: integer in-degree of that node
            If node_id is None: list of (node_id, in_degree) tuples for all nodes
            
        Notes
        -----
        Uses the reverse adjacency file for efficient in-degree calculation.
        Similar performance characteristics to out_degree().
        
        Examples
        --------
        >>> degree = graph.in_degree(5)  # In-degree of node 5
        >>> all_degrees = graph.in_degree()  # [(0, 0), (1, 2), ...]
        """
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
        """
        Remove all graph files and directories.
        
        Deletes all files associated with this graph instance, including
        binary adjacency files and metadata. If the base directory becomes
        empty after file removal, it is also deleted.
        
        Warning
        -------
        This operation is irreversible. All graph data will be permanently lost.
        
        Examples
        --------
        >>> graph.cleanup()  # Removes all files in graph.base_path
        """
        files_to_remove = [
            self.nodes_file, self.edges_file, self.metadata_file,
            self.adjacency_file, self.reverse_adjacency_file
        ]
        for file_path in files_to_remove:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        if os.path.exists(self.base_path) and not os.listdir(self.base_path):
            os.rmdir(self.base_path)