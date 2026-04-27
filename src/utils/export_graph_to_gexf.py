"""
Convert the undirected PyG similarity graph (.pt) to GEXF format.
"""
import os
import torch
import networkx as nx
from torch_geometric.data import Data
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert PyG graph to GEXF")
    parser.add_argument(
        "--input", type=str,
        default="results/similarity_graph/undirected_knn_graph.pt",
        help="Path to the .pt file"
    )
    parser.add_argument(
        "--output", type=str,
        default="results/similarity_graph/knn_graph.gexf",
        help="Output GEXF file path"
    )
    args = parser.parse_args()

    # 1. Load PyG Data
    data = torch.load(args.input)
    edge_index = data.edge_index
    node_dataset = data.node_dataset   # 0 = Tox21, 1 = NuBBE
    num_nodes = data.num_nodes

    # 2. Create NetworkX graph (undirected)
    G = nx.Graph()

    # Add nodes with attributes
    for i in range(num_nodes):
        if node_dataset[i] == 0:
            label = "HIV"
        else:
            label = "Antiviral"
        # Optionally include original index in the ID
        # node_id = f"{label}_{i}" if label == "NuBBE" else f"Tox21_{i}"
        node_id = f"{label}_{i}" if label == "Antiviral" else f"HIV_{i}"
        # or use plain integer ID if you prefer
        G.add_node(i, dataset=label, name=node_id)

    # 3. Add edges (edge_index is (2, E))
    for src, dst in edge_index.t().tolist():
        G.add_edge(src, dst)

    # 4. Write GEXF
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    nx.write_gexf(G, args.output)
    print(f"Graph exported to {args.output}")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    # print(f"  Tox21 nodes: {sum(1 for _, attr in G.nodes(data=True) if attr['dataset']=='Tox21')}")
    # print(f"  NuBBE nodes: {sum(1 for _, attr in G.nodes(data=True) if attr['dataset']=='NuBBE')}")
    print(f"  HIV nodes: {sum(1 for _, attr in G.nodes(data=True) if attr['dataset']=='HIV')}")
    print(f"  Antiviral nodes: {sum(1 for _, attr in G.nodes(data=True) if attr['dataset']=='Antiviral')}")

if __name__ == "__main__":
    main()