import os
import argparse
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch_geometric

from src.models.gin_model import GINEEncoder
from src.utils.graph_utils import extract_embeddings, build_threshold_similarity_graph


def load_graph_dataset(directory, label_transform=None):
    graphs = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith('.pt'):
            continue
        path = os.path.join(directory, fname)
        with torch.serialization.safe_globals([torch_geometric.data.Data]):
            data = torch.load(path, weights_only=False)
        if label_transform is not None:
            data.y = label_transform(data.y)
        graphs.append(data)
    return graphs


def encode_nubbe_label(y_str):
    if isinstance(y_str, str) and 'antioxidant' in y_str.lower():
        return torch.tensor([1.0])
    return torch.tensor([0.0])


def main():
    parser = argparse.ArgumentParser(description='Build a full threshold similarity graph without k-NN.')
    parser.add_argument('--tox21-dir', default='data/processed/graphs_tox21', help='Path to Tox21 graph .pt files')
    parser.add_argument('--nubbe-dir', default='data/processed/graphs_AntioxidantRos', help='Path to NuBBE graph .pt files')
    parser.add_argument('--encoder-checkpoint', default='models/checkpoints/tox21/best_encoder.pt', help='Path to pretrained encoder state dict')
    parser.add_argument('--threshold', type=float, default=0.95, help='Cosine similarity threshold for edge creation')
    parser.add_argument('--thresholds', nargs='+', type=float, help='Multiple cosine similarity thresholds for edge creation')
    parser.add_argument('--batch-size', type=int, default=32, help='DataLoader batch size')
    parser.add_argument('--save-dir', default='results/similiarity_graph', help='Directory to save the result graph and CSV')
    args = parser.parse_args()

    if args.thresholds:
        thresholds = args.thresholds
    else:
        thresholds = [args.threshold]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tox21_graphs = load_graph_dataset(args.tox21_dir)
    nubbe_graphs = load_graph_dataset(args.nubbe_dir, label_transform=encode_nubbe_label)

    print(f'Tox21 graphs: {len(tox21_graphs)}')
    print(f'NuBBE graphs: {len(nubbe_graphs)}')

    tox21_loader = DataLoader(tox21_graphs, batch_size=args.batch_size, shuffle=False)
    nubbe_loader = DataLoader(nubbe_graphs, batch_size=args.batch_size, shuffle=False)

    encoder = GINEEncoder(in_channels=35, edge_dim=9, hidden_channels=128, num_layers=4, dropout=0.2)
    encoder.load_state_dict(torch.load(args.encoder_checkpoint, map_location='cpu'))
    encoder.to(device).eval()

    print('Extracting embeddings...')
    tox21_emb = extract_embeddings(encoder, tox21_loader, device)
    nubbe_emb = extract_embeddings(encoder, nubbe_loader, device)
    all_emb = np.concatenate([tox21_emb, nubbe_emb], axis=0)
    num_nodes = all_emb.shape[0]

    print('Computing full pairwise similarities...')
    from sklearn.preprocessing import normalize
    emb_norm = normalize(all_emb, norm='l2')
    sim_matrix = emb_norm @ emb_norm.T
    i_upper, j_upper = np.triu_indices(num_nodes, k=1)
    sim_values = sim_matrix[i_upper, j_upper]

    os.makedirs(args.save_dir, exist_ok=True)
    all_sim_path = os.path.join(args.save_dir, 'all_pairwise_similarities.csv')
    with open(all_sim_path, 'w') as f:
        f.write('node1,node2,similarity\n')
        for i, j, sim in zip(i_upper, j_upper, sim_values):
            f.write(f'{i},{j},{sim:.6f}\n')
    print(f'Saved all pairwise similarities to {all_sim_path}')

    for thresh in thresholds:
        print(f'Building full threshold similarity graph for {num_nodes} nodes with threshold {thresh}')
        valid = sim_values >= thresh
        edge_similarities = {}
        if valid.any():
            i_valid = i_upper[valid]
            j_valid = j_upper[valid]
            sim_valid = sim_values[valid]

            for i, j, sim in zip(i_valid, j_valid, sim_valid):
                edge_similarities[(int(i), int(j))] = float(sim)

            edge_list = np.vstack([np.concatenate([i_valid, j_valid]), np.concatenate([j_valid, i_valid])])
            edge_index = torch.tensor(edge_list, dtype=torch.long)
            edge_index = torch.unique(edge_index, dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_similarities = {}

        similarity_path = os.path.join(args.save_dir, f'edge_similarities_threshold_{thresh}.csv')
        df = pd.DataFrame([
            {'node1': i, 'node2': j, 'similarity': sim}
            for (i, j), sim in edge_similarities.items()
        ])
        df.to_csv(similarity_path, index=False)
        print(f'Saved edge similarities to {similarity_path}')

        mask = edge_index[0] < edge_index[1]
        undirected_edge_index = edge_index[:, mask]
        graph = Data(edge_index=undirected_edge_index, num_nodes=num_nodes)
        graph_path = os.path.join(args.save_dir, f'undirected_threshold_{thresh}_graph.pt')
        torch.save(graph, graph_path)
        print(f'Saved undirected threshold graph to {graph_path}')
        print(f'Graph statistics: {num_nodes} nodes, {undirected_edge_index.shape[1]} undirected edges')


if __name__ == '__main__':
    main()
