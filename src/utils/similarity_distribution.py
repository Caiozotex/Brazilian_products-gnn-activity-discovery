"""
Distribuição dos valores de similaridade cosseno 2 a 2 entre embeddings de moléculas.

Carrega o encoder pré-treinado, extrai embeddings do dataset escolhido e computa
todas as similaridades par-a-par (i, j) com i < j. Salva histograma em
results/similarity_distribution/.

NOTA: o nubbe-env tem o matplotlib com DLL quebrada no Windows. Execute este script
com o Python do ambiente base do Anaconda:
    C:\\Users\\davic\\anaconda3\\python.exe -c "..." (ver README ou run_sim_dist.py)

Fluxo recomendado:
  1. Extrair embeddings com nubbe-env  → salva results/similarity_distribution/embeddings.npy
  2. Plotar com Python base            → lê o .npy e gera o PNG
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from torch_geometric.loader import DataLoader

from src.models.gin_model import GINEEncoder
from src.utils.graph_utils import extract_embeddings

# ── Defaults ──────────────────────────────────────────────────────────────────
ENCODER_PATH = "models/checkpoints/nubbe_antioxidant/best_encoder.pt"
PROCESSED_DIR = "data/processed/graphs_AntioxidantRos"
OUTPUT_DIR = "results/similarity_distribution"
BATCH_SIZE = 64
IN_CHANNELS = 35
EDGE_DIM = 9
HIDDEN_CHANNELS = 128
NUM_LAYERS = 4


def load_graphs(processed_dir: str):
    """Carrega todos os arquivos .pt do diretório processado."""
    paths = [
        os.path.join(processed_dir, f)
        for f in os.listdir(processed_dir)
        if f.endswith(".pt")
    ]
    if not paths:
        raise FileNotFoundError(f"Nenhum .pt encontrado em {processed_dir}")
    graphs = [torch.load(p, weights_only=False) for p in paths]
    return graphs


def compute_all_pairwise_cosine(embeddings: np.ndarray) -> np.ndarray:
    """
    Calcula similaridade cosseno para todos os pares (i, j) com i < j.
    Retorna array 1-D com os valores em [0, 1].
    """
    emb = normalize(embeddings, norm="l2")       # (N, D)
    dot = emb @ emb.T                             # (N, N) — similaridades cosseno
    n = emb.shape[0]
    # Extrai triângulo superior sem a diagonal
    idx = np.triu_indices(n, k=1)
    similarities = dot[idx]
    # Clipa para [0, 1] (erros numéricos podem gerar valores fora do intervalo)
    similarities = np.clip(similarities, 0.0, 1.0)
    return similarities


def plot_distribution(similarities: np.ndarray, output_dir: str, dataset_name: str):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Distribuição de Similaridade Cosseno 2 a 2\n"
        f"Dataset: {dataset_name}  |  {len(similarities):,} pares",
        fontsize=13,
    )

    # ── Histograma ────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.hist(similarities, bins=100, range=(0, 1), color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Similaridade cosseno", fontsize=11)
    ax.set_ylabel("Nº de pares", fontsize=11)
    ax.set_title("Histograma", fontsize=11)
    ax.set_xlim(0, 1)
    ax.axvline(similarities.mean(), color="tomato", linestyle="--", linewidth=1.5,
               label=f"Média = {similarities.mean():.3f}")
    ax.axvline(np.median(similarities), color="gold", linestyle="--", linewidth=1.5,
               label=f"Mediana = {np.median(similarities):.3f}")
    ax.legend(fontsize=9)

    # ── CDF ───────────────────────────────────────────────────────────────────
    ax2 = axes[1]
    sorted_sim = np.sort(similarities)
    cdf = np.arange(1, len(sorted_sim) + 1) / len(sorted_sim)
    ax2.plot(sorted_sim, cdf, color="#4C72B0", linewidth=1.5)
    ax2.set_xlabel("Similaridade cosseno", fontsize=11)
    ax2.set_ylabel("Fração acumulada de pares", fontsize=11)
    ax2.set_title("CDF", fontsize=11)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{dataset_name}_similarity_distribution.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo em: {out_path}")
    return out_path


def print_stats(similarities: np.ndarray):
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("\n── Estatísticas de similaridade cosseno 2 a 2 ──")
    print(f"  Total de pares : {len(similarities):,}")
    print(f"  Mínimo         : {similarities.min():.4f}")
    print(f"  Máximo         : {similarities.max():.4f}")
    print(f"  Média          : {similarities.mean():.4f}")
    print(f"  Desvio padrão  : {similarities.std():.4f}")
    for p in percentiles:
        print(f"  P{p:02d}            : {np.percentile(similarities, p):.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Distribuição de similaridade 2 a 2")
    parser.add_argument("--encoder", default=ENCODER_PATH, help="Caminho para best_encoder.pt")
    parser.add_argument("--processed_dir", default=PROCESSED_DIR, help="Dir com grafos .pt")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Dir de saída do gráfico")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dataset_name", default="nubbe_antioxidant",
                        help="Nome do dataset (usado no título e no nome do arquivo)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # ── Carrega grafos ────────────────────────────────────────────────────────
    print(f"Carregando grafos de: {args.processed_dir}")
    graphs = load_graphs(args.processed_dir)
    print(f"  {len(graphs)} moléculas carregadas")

    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)

    # ── Carrega encoder ───────────────────────────────────────────────────────
    encoder = GINEEncoder(
        in_channels=IN_CHANNELS,
        edge_dim=EDGE_DIM,
        hidden_channels=HIDDEN_CHANNELS,
        num_layers=NUM_LAYERS,
    ).to(device)

    state = torch.load(args.encoder, map_location=device, weights_only=False)
    # Aceita tanto state_dict puro quanto checkpoint com chave "encoder_state_dict"
    if isinstance(state, dict) and "encoder_state_dict" in state:
        state = state["encoder_state_dict"]
    encoder.load_state_dict(state)
    encoder.eval()
    print(f"Encoder carregado de: {args.encoder}")

    # ── Extrai embeddings ─────────────────────────────────────────────────────
    print("Extraindo embeddings...")
    embeddings = extract_embeddings(encoder, loader, device)
    print(f"  Shape: {embeddings.shape}")

    # ── Similaridades 2 a 2 ───────────────────────────────────────────────────
    print("Calculando similaridades 2 a 2 (pode demorar para N grande)...")
    similarities = compute_all_pairwise_cosine(embeddings)

    print_stats(similarities)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_distribution(similarities, args.output_dir, args.dataset_name)


if __name__ == "__main__":
    main()
