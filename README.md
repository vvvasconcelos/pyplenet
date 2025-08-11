# Pyplenet

A high-performance network generation tool optimized for large-scale social networks. Pyplenet creates realistic social networks using preferential attachment + reciprocity algorithm and stores them efficiently using file-based adjacency lists, enabling the generation of networks with millions of nodes and edges.

## Key Features

- **Memory-Efficient**: File-based adjacency list storage handles networks larger than RAM
- **High Performance**: Optimized algorithms for million+ node networks  
- **Multiple Export Formats**: Export to (most?) standard graph formats (MTX, SNAP, EdgeList, etc.)

## WIP Project Structure

```
pyplenet/
├── generate.py         # Main network generation script
├── graph.py            # File-based graph implementation  
├── grn.py              # Graph/network generation algorithms
├── netstats.py         # Network statistics and analysis
├── graph_export.py     # Export to multiple formats
├── graph_inspect.py    # Graph inspection utilities
├── utils.py            # Data processing utilities
└── demo_adjacency.py   # Example usage
```

## Core Components

### FileBasedGraph (`graph.py`)
The core graph data structure optimized for large networks:
- **Binary file storage**: Adjacency lists stored as binary files
- **Memory mapping**: Efficient access to large files
- **Batch processing**: Processes edges in chunks for scalability
- **Metadata tracking**: Node attributes and graph properties

### Network Generation (`generate.py`)
Main script for creating social networks:
- **Population-based**: Creates nodes from demographic data
- **Link generation**: Connects nodes using household/social relationship data
- **Preferential attachment**: Realistic network topology
- **Scalability**: Scale parameter controls network size

### Graph Export (`graph_export.py`)
Export networks to standard formats:
- **Binary edge lists**: Most compact format (8 bytes per edge)
- **Text formats**: Human-readable edge lists and adjacency lists
- **Matrix Market (.mtx)**: Standard sparse matrix format
- **SNAP format**: Stanford Network Analysis Platform format
- **CSV**: Spreadsheet-compatible format

## Quick Start

### Basic Usage

```python
from generate import generate

# Generate a small network
graph = generate(
    pops_path='Data/population.xlsx',     # Population data
    links_path='Data/households.xlsx',    # Relationship data  
    fraction=0.4,                         # Preferential attachment fraction
    scale=0.1,                           # Scale factor (0.1 = 10% of full size)
    reciprocity_p=0.8                    # Reciprocal edge probability
)

print(f"Generated network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
```

### Export to Multiple Formats

```python
from graph_export import export_all_formats

# Export to all supported formats
export_all_formats(graph, "my_network")

# Creates:
# - my_network_edges.txt (text edge list)
# - my_network_edges.bin (binary edge list) 
# - my_network.mtx (Matrix Market)
# - my_network.snap (SNAP format)
# - my_network_edges.csv (CSV format)
```

### Large-Scale Generation

```python
# Generate a large network (file-based storage)
large_graph = generate(
    pops_path='Data/population.xlsx',
    links_path='Data/households.xlsx', 
    fraction=0.4,
    scale=1.0,                           # Full scale
    reciprocity_p=0.8,
    base_path="large_network_data"       # Custom storage location
)

# Network data stored in large_network_data/
# - adjacency.dat (forward edges)
# - reverse_adjacency.dat (backward edges)
# - metadata.json (graph properties)
```

## Network Statistics

NOTE: Not (yet) optimized for larger networks.
Calculate network properties:

```python
from netstats import runstats

# Comprehensive network analysis
stats = runstats(graph)

# Individual statistics
reciprocity = calculate_reciprocity(graph)
components = connected_components(graph) 
clustering = average_clustering(graph)
```

## Performance

Pyplenet is optimized for large-scale networks:

- **Memory Usage**: O(1) RAM memory usage regardless of network size
- **Generation Speed**: ~100K edges/second on modern hardware
- **File Size**: ~56MB per million edges in binary format
- **Scalability**: Successfully tested with 1M+ nodes, 10M+ edges

### Performance Comparison

| Format | Size (1M edges) | Compression Ratio |
|--------|----------------|-------------------|
| Binary | 8 MB | 1x (baseline) |
| Text | 20 MB | 2.5x |
| Gzipped Text | 5 MB | 0.6x |
| CSV | 25 MB | 3.1x |

## WIP Data Requirements

### Population Data Format
Excel/CSV file with demographic characteristics:

```csv
age,gender,education,n
20,M,high_school,850
20,F,high_school,900
25,M,college,1200
25,F,college,1100
30,M,graduate,800
30,F,graduate,750
35,M,high_school,650
35,F,college,700
...
```

### Link Data Format
Excel/CSV file defining number of edges between two separate groups:
TODO: make _src and _dst suffixes customizable

```csv
age_src,gender_src,education_src,age_dst,gender_dst,education_dst,n
20,M,high_school,20,F,high_school,450
25,M,college,25,F,college,380
30,M,graduate,28,F,college,220
25,F,college,30,M,graduate,180
35,M,high_school,32,F,high_school,150
...
```

## Advanced Usage

### Custom Graph Inspection

```python
from graph_inspect import inspect_graph

# Analyze graph structure  
inspect_graph("graph_data/")

# Check specific nodes
node_info = graph.get_node_attributes(node_id)
out_edges = list(graph.get_out_edges(node_id))
in_edges = list(graph.get_in_edges(node_id))
```

### Batch Processing

```python
# Process large networks in batches
for batch_nodes in batch_iterator(graph.nodes(), batch_size=10000):
    # Process batch of nodes
    process_node_batch(batch_nodes)
```

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib (for statistics/plotting)

## Installation

```bash
# Clone repository
git clone https://github.com/matijsv/pyplenet.git
cd pyplenet

# Install dependencies
pip install numpy pandas matplotlib openpyxl
```