import pandas as pd

def find_nodes(G, **attrs):
    attrs_key = tuple(sorted(attrs.items()))
    
    group_id = G.attrs_to_group[attrs_key]
    if group_id is None:
        return []
    list_of_nodes = G.group_to_nodes[group_id]
    #print(attrs_key)
    #print(list_of_nodes[0], list_of_nodes[-1])
    return list_of_nodes, group_id

def read_file(path):
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.xlsx'):
        return pd.read_excel(path)
    else:
        raise ValueError("Unsupported file format: {}".format(path))

def desc_groups(pops_path):
    df_group_pops = read_file(pops_path)

    # Identify characteristic columns (all except 'n')
    characteristic_cols = [col for col in df_group_pops.columns if col != 'n']

    # Each group gets a unique ID (e.g., row number)
    group_populations = {
        idx: {**{col: row[col] for col in characteristic_cols}, 'n': row['n']}
        for idx, row in df_group_pops.iterrows()
    }

    return group_populations, characteristic_cols

import csv

def export_group_to_attrs_csv(group_to_attrs, filename):
    """Export group_to_attrs dict to CSV."""
    # Get all attribute keys
    all_keys = set()
    for attrs in group_to_attrs.values():
        all_keys.update(attrs.keys())
    fieldnames = ['group_id'] + sorted(all_keys)
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for group_id, attrs in group_to_attrs.items():
            row = {'group_id': group_id}
            row.update(attrs)
            writer.writerow(row)

def export_group_to_nodes_csv(group_to_nodes, filename):
    """Export group_to_nodes dict to CSV with start and end node_id."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['group_id', 'start_node_id', 'end_node_id'])
        for group_id, node_ids in group_to_nodes.items():
            if node_ids:
                writer.writerow([group_id, node_ids[0], node_ids[-1]])