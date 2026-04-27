import os
import torch
from torch_geometric.loader import DataLoader
import numpy as np
from torch_geometric.data import Data

from src.models.gin_model import GINEEncoder
from src.models.task_models import Tox21Model, NuBBEModel, HIVModel
from src.train.pretrain import pretrain_tox21, pretrain_hiv
from src.train.finetune import finetune_nubbe
from src.utils.graph_utils import extract_embeddings,build_knn_similarity_graph_faiss


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Directories where individual .pt files live
    # TOX21_DIR = "data/processed/graphs_tox21"
    # ANTIOX_DIR = "data/processed/graphs_AntioxidantRos"
    HIV_DIR = "data/processed/graphs_hiv"
    ANTIVIRAL_DIR = "data/processed/graphs_brnpdb_antiviral"

    # # -------------------------------------------------------
    # # Load ALL graphs into a list
    # # -------------------------------------------------------
    # # Tox21
    # tox21_list = []
    # for fname in sorted(os.listdir(TOX21_DIR)):
    #     if fname.endswith('.pt'):
    #         data = torch.load(os.path.join(TOX21_DIR, fname))
    #         # Ensure y is 1‑D with 12 elements
    #         data.y = data.y.view(12)          # reshape to (12,)
    #         tox21_list.append(data)

    # # NuBBE (AntioxidantRos)
    # def encode_label(y_str):
    #     """Convert bioactivity string to binary tensor."""
    #     if isinstance(y_str, str) and "antioxidant" in y_str.lower():
    #         return torch.tensor([1.0])
    #     else:
    #         return torch.tensor([0.0])
    
    # nubbe_list = []
    # for fname in sorted(os.listdir(ANTIOX_DIR)):
    #     if fname.endswith('.pt'):
    #         data = torch.load(os.path.join(ANTIOX_DIR, fname))
    #         # Convert string label to binary tensor (shape (1,))
    #         data.y = encode_label(data.y)
    #         nubbe_list.append(data)

    # print(f"Tox21 graphs: {len(tox21_list)}")
    # print(f"NuBBE (AntioxidantRos) graphs: {len(nubbe_list)}")

    # HIV dataset (binary classification: 1 = active, 0 = inactive)
    hiv_list = []
    for fname in sorted(os.listdir(HIV_DIR)):
        if fname.endswith('.pt'):
            data = torch.load(os.path.join(HIV_DIR, fname))
            # Ensure y is a scalar tensor (binary label)
            # HIVDataset stores binary label in data.y (shape (1,)) or data.hiv_active
            if hasattr(data, 'hiv_active'):
                data.y = data.hiv_active  # already a tensor
            elif hasattr(data, 'y') and data.y.numel() == 1:
                pass  # already single value
            elif hasattr(data, 'y') and data.y.numel() > 1:
                data.y = data.y[0]  # take first element if multi-label
            else:
                raise ValueError(f"HIV graph {fname} missing label")
            # Ensure shape (1,) for consistency
            data.y = data.y.view(1)
            hiv_list.append(data)

    # BrNPDB Antiviral dataset (all compounds are antiviral, so label = 1.0)
    antiviral_list = []
    for fname in sorted(os.listdir(ANTIVIRAL_DIR)):
        if fname.endswith('.pt'):
            data = torch.load(os.path.join(ANTIVIRAL_DIR, fname))
            # Assign a fixed label (1 = antiviral) – useful for pretraining or as positive set
            data.y = torch.tensor([1.0])
            # Optional: store the original brnpdb_id or common_name if needed
            # if hasattr(data, 'brnpdb_id'): ...
            antiviral_list.append(data)

    print(f"HIV graphs: {len(hiv_list)}")
    print(f"BrNPDB Antiviral graphs: {len(antiviral_list)}")


    # -------------------------------------------------------
    # Create DataLoaders
    # -------------------------------------------------------
    # tox21_loader = DataLoader(tox21_list, batch_size=32, shuffle=True)
    # nubbe_loader = DataLoader(nubbe_list, batch_size=32, shuffle=True)
    hiv_loader = DataLoader(hiv_list, batch_size=32, shuffle=True)
    antiviral_loader = DataLoader(antiviral_list, batch_size=32, shuffle=True)


    # # dimensions
    num_node_feats = 35      
    num_edge_feats = 9      

    # 1. Build encoder
    encoder = GINEEncoder(
        in_channels=num_node_feats,
        edge_dim=num_edge_feats,
        hidden_channels=128,
        num_layers=4,
        dropout=0.2
    ).to(device)



    # tox21_model = Tox21Model(encoder, hidden_dim=128, num_tasks=12, dropout_head=0.2).to(device)
    # optimizer = torch.optim.Adam(tox21_model.parameters(), lr=1e-3)
    # hiv_model = HIVModel(encoder, hidden_dim=128, num_tasks=1, dropout_head=0.2).to(device)
    # optimizer = torch.optim.Adam(hiv_model.parameters(), lr=1e-3)

    

    # Pretrain HIV with early stopping and checkpoints
    # hiv_model = pretrain_hiv(
    #     model=hiv_model,
    #     train_loader=hiv_loader,   
    #     optimizer=optimizer,
    #     device=device,
    #     val_loader=None,             
    #     epochs=100,
    #     save_dir="models/checkpoints/hiv",
    #     patience=10,
    #     checkpoint_every=25
    # )


    # # Pretrain Tox21  with early stopping and checkpoints
    # tox21_model = pretrain_tox21(
    #     model=tox21_model,
    #     train_loader=tox21_loader,   
    #     optimizer=optimizer,
    #     device=device,
    #     val_loader=None,             
    #     epochs=500,
    #     save_dir="models/checkpoints/tox21",
    #     patience=50,
    #     checkpoint_every=50
    # )

    # Load best encoder (no head)
    # encoder = GINEEncoder(in_channels=num_node_feats,edge_dim=num_edge_feats, hidden_channels=128, num_layers=4, dropout=0.2).to(device)
    # encoder.load_state_dict(torch.load("models/checkpoints/tox21/best_encoder.pt"))
    # encoder.eval()

    # Load best encoder (no head)
    # encoder = GINEEncoder(in_channels=num_node_feats,edge_dim=num_edge_feats, hidden_channels=128, num_layers=4, dropout=0.2).to(device)
    encoder.load_state_dict(torch.load("models/checkpoints/hiv/best_encoder.pt"))
    encoder.eval()



    # 2. Build NuBBE model with new head
    # nubbe_model = NuBBEModel(encoder, hidden_dim=128, dropout_head=0.2).to(device)

    # # 3. Optimizer (low learning rate for fine‑tuning)
    # optimizer = torch.optim.Adam(nubbe_model.parameters(), lr=1e-4)

    # # 4. Fine‑tune on NuBBE
    # nubbe_model = finetune_nubbe(
    #     model=nubbe_model,
    #     train_loader=nubbe_loader,  
    #     optimizer=optimizer,
    #     device=device,
    #     val_loader=None,             
    #     epochs=100,
    #     save_dir="models/checkpoints/nubbe_antioxidant",
    #     patience=10,
    #     checkpoint_every=50
    # )

    # Option A: Tox21‑pretrained encoder
    # encoder = GINEEncoder(in_channels=num_node_feats,edge_dim=num_edge_feats, hidden_channels=128, num_layers=4, dropout=0.2).to(device)
    # encoder.load_state_dict(torch.load("models/checkpoints/tox21/best_encoder.pt", map_location="cpu"))

    # # Option B: NuBBE fine‑tuned encoder (if you fine‑tuned)
    # # encoder.load_state_dict(torch.load("models/checkpoints/nubbe_antioxidant/best_encoder.pt", map_location="cpu"))

    # encoder.eval()

    # # Move to GPU if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # encoder.to(device)



    # # # 4. Extract embeddings for all molecules
    # tox21_emb   = extract_embeddings(encoder, tox21_loader, device)
    # nubbe_emb   = extract_embeddings(encoder, nubbe_loader, device)
    # assert np.all(np.isfinite(tox21_emb)), "Tox21 embeddings contain NaN or Inf"
    # assert np.all(np.isfinite(nubbe_emb)), "NuBBE embeddings contain NaN or Inf"
    # n_tox = tox21_emb.shape[0]          # number of Tox21 molecules
    # n_nub = nubbe_emb.shape[0]          # number of NuBBE molecules
    # all_emb = np.concatenate([tox21_emb, nubbe_emb], axis=0)


    # # # 4. Extract embeddings for all molecules
    hiv_emb   = extract_embeddings(encoder, hiv_loader, device)
    antiviral_emb   = extract_embeddings(encoder, antiviral_loader, device)
    assert np.all(np.isfinite(hiv_emb)), "HIV embeddings contain NaN or Inf"
    assert np.all(np.isfinite(antiviral_emb)), "NuBBE embeddings contain NaN or Inf"
    n_hiv = hiv_emb.shape[0]          # number of HIV molecules
    #print(n_hiv)
    n_antiviral = antiviral_emb.shape[0]    # number of NuBBE molecules
    #print(n_antiviral)
    all_emb = np.concatenate([hiv_emb, antiviral_emb], axis=0)
 

    # # # 5. Build k-NN similarity graph
    # edge_index, num_nodes = build_knn_similarity_graph_faiss(all_emb, k=10, threshold=0.6)
    # print(f"Graph built: {num_nodes} nodes")

    # # Remove duplicate undirected edges (keep i < j)
    # mask = edge_index[0] < edge_index[1]
    # undirected_edge_index = edge_index[:, mask]

    # # 2. Optional: verify it's roughly half
    # print(f"Directed edges (both dirs): {edge_index.shape[1]}")
    # print(f"Undirected edges (unique): {undirected_edge_index.shape[1]}")

    # # # # Create a node attribute that indicates the dataset:
    # # # # 0 = Tox21, 1 = NuBBE
    # node_dataset = torch.zeros(num_nodes, dtype=torch.long)
    # node_dataset[n_tox : n_tox + n_nub] = 1   # indices after Tox21 are NuBBE

    # # # # The resulting graph can be wrapped as a PyG Data object:
    # similarity_graph_undirected = Data(
    # edge_index=undirected_edge_index,
    # num_nodes=num_nodes,
    # node_dataset=node_dataset)

    # save_dir = "results/similiarity_graph"
    # save_path = os.path.join(save_dir, "undirected_knn_graph.pt")
    # torch.save(similarity_graph_undirected, save_path)
    # print(f"Undirected graph saved to {save_path}")


    # # # 5. Build k-NN similarity graph
    edge_index, num_nodes = build_knn_similarity_graph_faiss(all_emb, k=10, threshold=0.9)
    print(f"Graph built: {num_nodes} nodes")

    # Remove duplicate undirected edges (keep i < j)
    mask = edge_index[0] < edge_index[1]
    undirected_edge_index = edge_index[:, mask]

    # 2. Optional: verify it's roughly half
    print(f"Directed edges (both dirs): {edge_index.shape[1]}")
    print(f"Undirected edges (unique): {undirected_edge_index.shape[1]}")

    # # # Create a node attribute that indicates the dataset:
    # # # 0 = HIV, 1 = Antiviral
    node_dataset = torch.zeros(num_nodes, dtype=torch.long)
    node_dataset[n_hiv : n_hiv + n_antiviral] = 1   # indices after Tox21 are NuBBE

    # # # The resulting graph can be wrapped as a PyG Data object:
    similarity_graph_undirected = Data(
    edge_index=undirected_edge_index,
    num_nodes=num_nodes,
    node_dataset=node_dataset)



    save_dir = "results/similiarity_graph_hiv"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "undirected_graph_hiv.pt")
    torch.save(similarity_graph_undirected, save_path)
    print(f"Undirected graph saved to {save_path}")

