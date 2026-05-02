import os
import torch
from torch_geometric.data import Data

# Paths to the processed datasets
TOX21_DIR = "data/processed/graphs_tox21"
ANTIOX_DIR = "data/processed/graphs_AntioxidantRos"

# Expected feature sizes (from your first inspection)
EXPECTED_NODE_FEATS = 35
EXPECTED_EDGE_FEATS = 9

def inspect_dataset(path, name):
    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    print(f"Path: {path}")
    print(f"{'='*50}")


    data_file = None

    for f in os.listdir(path):
        if f.endswith('.pt'):
            data_file = os.path.join(path, f)
            break
 

    print(f"Loading from: {data_file}")
    data = torch.load(data_file)

    # The structure can be a list of Data objects, a single Data object,
    # or a tuple (data_list, slices) if saved via InMemoryDataset.save().
    if isinstance(data, (list, tuple)):
        if len(data) == 2 and isinstance(data[0], list) and isinstance(data[1], dict):
            # Typical PyG InMemoryDataset output: (data_list, slices)
            graph_list = data[0]
        else:
            # Assume it's a list of Data
            graph_list = data
    elif hasattr(data, 'x'):   # single Data object
        graph_list = [data]
    else:
        print("ERROR: Unrecognised data format.")
        return

    # Inspect the first graph
    sample = graph_list[0]
    if not hasattr(sample, 'x') or sample.x is None:
        print("ERROR: First graph has no node features (x).")
    else:
        num_node_feats = sample.x.size(1)
        print(f"Number of node features (x.shape[1]): {num_node_feats}")

    if not hasattr(sample, 'edge_attr') or sample.edge_attr is None:
        print("WARNING: First graph has no edge features (edge_attr).")
        num_edge_feats = None
    else:
        num_edge_feats = sample.edge_attr.size(1)
        print(f"Number of edge features (edge_attr.shape[1]): {num_edge_feats}")

    print(f"Total graphs loaded: {len(graph_list)}")

def validate_dataset(directory):
    print(f"Scanning directory: {directory}")
    pt_files = [f for f in os.listdir(directory) if f.endswith('.pt')]
    print(f"Found {len(pt_files)} .pt file(s)\n")

    if not pt_files:
        print("No .pt files found.")
        return

    valid = 0
    invalid = 0
    errors = []

    first_node_feats = None
    first_edge_feats = None

    for fname in pt_files:
        path = os.path.join(directory, fname)
        try:
            data = torch.load(path)
        except Exception as e:
            errors.append(f"{fname}: failed to load ({e})")
            invalid += 1
            continue

        # Check that it's a PyG Data object
        if not isinstance(data, Data):
            errors.append(f"{fname}: not a torch_geometric.data.Data object")
            invalid += 1
            continue

        # Check essential attributes
        for attr in ['x', 'edge_index']:
            if not hasattr(data, attr) or getattr(data, attr) is None:
                errors.append(f"{fname}: missing required attribute '{attr}'")
                invalid += 1
                continue

        # Node features
        if data.x.dim() != 2:
            errors.append(f"{fname}: x must be 2-dimensional (got {data.x.dim()})")
            invalid += 1
            continue
        node_feats = data.x.size(1)
        if node_feats != EXPECTED_NODE_FEATS:
            errors.append(
                f"{fname}: x.shape[1] = {node_feats}, expected {EXPECTED_NODE_FEATS}"
            )
            invalid += 1
            continue

        # Edge index
        if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
            errors.append(f"{fname}: edge_index must be (2, E)")
            invalid += 1
            continue

        # Edge attributes
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            if data.edge_attr.dim() != 2:
                errors.append(f"{fname}: edge_attr must be 2-dimensional")
                invalid += 1
                continue
            edge_feats = data.edge_attr.size(1)
            if edge_feats != EXPECTED_EDGE_FEATS:
                errors.append(
                    f"{fname}: edge_attr.shape[1] = {edge_feats}, expected {EXPECTED_EDGE_FEATS}"
                )
                invalid += 1
                continue
        else:
            # If edge_attr is missing - is that allowed? Your earlier inspection found it.
            errors.append(f"{fname}: edge_attr is missing or None")
            invalid += 1
            continue

        # Check y (label) exists, but don't enforce shape necessarily
        if not hasattr(data, 'y') or data.y is None:
            errors.append(f"{fname}: missing label 'y'")
            invalid += 1
            continue

        # Optional: print if y shape differs from first file (just a warning)
        if first_node_feats is None:
            first_node_feats = node_feats
            first_edge_feats = edge_feats

        valid += 1

    print(f"Validation complete: {valid} valid, {invalid} invalid")
    if errors:
        print("Errors/Warnings:")
        for e in errors:
            print("  -", e)
    else:
        print("All files passed validation.")

if __name__ == "__main__":
    print("Inspecting Tox21 and AntioxidantRos datasets...")
    inspect_dataset(TOX21_DIR, "Tox21")
    inspect_dataset(ANTIOX_DIR, "AntioxidantRos")
    print("\nDone.")
    validate_dataset(ANTIOX_DIR)
    validate_dataset(TOX21_DIR)