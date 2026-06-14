
import os
import torch
import networkx as nx
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert PyG graph to GEXF")
    parser.add_argument(
        "--input", type=str,
        default="results/similiarity_graph_hiv/undirected_graph_hiv.pt",
        help="Path to the .pt file"
    )
    parser.add_argument(
        "--output", type=str,
        default="results/similiarity_graph_hiv/hiv_knn_graph.gexf",
        help="Output GEXF file path"
    )
    args = parser.parse_args()
    # -------------------------------------------------------
    # 1. Load PyG Data
    # -------------------------------------------------------
    data = torch.load(args.input)

    edge_index = data.edge_index
    num_nodes = data.num_nodes

    # Load attributes safely
    node_dataset = data.node_dataset if hasattr(data, "node_dataset") else None
    hiv_active = data.hiv_active if hasattr(data, "hiv_active") else None
    edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
    brnpdb_ids = data.brnpdb_id if hasattr(data, "brnpdb_id") else None
    common_names = data.common_name if hasattr(data, "common_name") else None

    # -------------------------------------------------------
    # 2. Create NetworkX graph
    # -------------------------------------------------------
    G = nx.Graph()

    # -------------------------------------------------------
    # 3. Add nodes with attributes
    # -------------------------------------------------------
    for i in range(num_nodes):

        # Dataset label
        if node_dataset is not None:
            dataset_label = "HIV" if node_dataset[i] == 0 else "Antiviral"
        else:
            dataset_label = "Unknown"

        # HIV activity label
        if hiv_active is not None:
            hiv_val = float(hiv_active[i].item())
        else:
            hiv_val = None

        # Visualization label type
        if hiv_val == -1:
            label_type = "unlabeled"
        elif hiv_val == 1:
            label_type = "active"
        elif hiv_val == 0:
            label_type = "inactive"
        else:
            label_type = "unknown"

        node_name = f"{dataset_label}_{i}"

        node_attrs = {
            "dataset": dataset_label,
            "hiv_active": hiv_val,
            "label_type": label_type,
            "name": node_name
        }

        # ---------------------------------------------------
        # Add BrNPDB metadata for antiviral nodes
        # ---------------------------------------------------

        if dataset_label == "Antiviral":

            if brnpdb_ids is not None:
                node_attrs["brnpdb_id"] = int(brnpdb_ids[i])

            if common_names is not None:
                node_attrs["common_name"] = str(common_names[i])

        G.add_node(i, **node_attrs)
            

    # -------------------------------------------------------
    # 4. Add edges (with optional weights)
    # -------------------------------------------------------
    for idx, (src, dst) in enumerate(edge_index.t().tolist()):
        if edge_attr is not None:
            weight = float(edge_attr[idx].item())
            G.add_edge(src, dst, weight=weight)
        else:
            G.add_edge(src, dst)

    # -------------------------------------------------------
    # 5. Save GEXF
    # -------------------------------------------------------
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    nx.write_gexf(G, args.output)

    # -------------------------------------------------------
    # 6. Debug / sanity prints
    # -------------------------------------------------------
    print(f"Graph exported to {args.output}")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    print(f"HIV nodes: {sum(1 for _, d in G.nodes(data=True) if d['dataset'] == 'HIV')}")
    print(f"Antiviral nodes: {sum(1 for _, d in G.nodes(data=True) if d['dataset'] == 'Antiviral')}")

    if hiv_active is not None:
        print(f"Active nodes: {sum(1 for _, d in G.nodes(data=True) if d['hiv_active'] == 1)}")
        print(f"Inactive nodes: {sum(1 for _, d in G.nodes(data=True) if d['hiv_active'] == 0)}")
        print(f"Unlabeled nodes: {sum(1 for _, d in G.nodes(data=True) if d['hiv_active'] == -1)}")


if __name__ == "__main__":
    main()
