"""
Compute graph metrics and visualisations for a single GEXF graph.

"""

import os
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute metrics and plots for a GEXF graph."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the .gexf file."
    )
    parser.add_argument(
        "--output_base", type=str, default="results",
        help="Base directory for results (default: results)."
    )
    return parser.parse_args()


def save_metrics_txt(G, filepath):
    """Write basic graph metrics to a text file."""
    n_nodes = nx.number_of_nodes(G)
    n_edges = nx.number_of_edges(G)
    degrees = dict(G.degree())
    avg_degree = np.mean(list(degrees.values()))
    n_components = nx.number_connected_components(G)

    comp_sizes = sorted(
        [len(c) for c in nx.connected_components(G)], reverse=True
    )

    with open(filepath, "w") as f:
        f.write(f"Number of vertices: {n_nodes}\n")
        f.write(f"Number of edges: {n_edges}\n")
        f.write(f"Average degree: {avg_degree:.2f}\n")
        f.write(f"Number of connected components: {n_components}\n")
        f.write(f"Component sizes (sorted): {comp_sizes}\n")

    # Also print to console for quick inspection
    print(f"Vertices: {n_nodes}")
    print(f"Edges: {n_edges}")
    print(f"Avg degree: {avg_degree:.2f}")
    print(f"Components: {n_components}")


def plot_graph_figure(G, figure_path):
    """
    Generate a figure with two subplots:
    - Degree histogram (linear scale)
    - Component size distribution (bar chart)
    """
    # Basic properties
    components = list(nx.connected_components(G))
    comp_sizes = [len(c) for c in components]
    n_components = len(components)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Degree histogram ----
    degrees = [d for _, d in G.degree()]
    max_degree = max(degrees) if degrees else 0
    bins = np.arange(0, max_degree + 2) - 0.5  # center bars on integers
    ax1.hist(degrees, bins=bins, edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Degree")
    ax1.set_ylabel("Frequency (number of nodes)")
    ax1.set_title("Degree Distribution")
    ax1.grid(axis="y", alpha=0.5)

    # ---- Component size histogram ----
    if n_components > 1:
        sizes, freq = np.unique(comp_sizes, return_counts=True)
        ax2.bar(sizes, freq, width=1.0, edgecolor="black")
        ax2.set_xlabel("Component size (number of vertices)")
        ax2.set_ylabel("Number of components")
        ax2.set_title("Component Size Distribution")
        ax2.grid(axis="y", alpha=0.5)
        # If there are many component sizes, maybe use log scale? We'll keep linear for clarity.
    else:
        ax2.text(0.5, 0.5, "Only one connected component",
                 transform=ax2.transAxes, ha="center", fontsize=12)
        ax2.set_title("Component Size Distribution")
        ax2.axis("off")

    plt.tight_layout()
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {figure_path}")


def main():
    args = parse_args()
    gexf_path = Path(args.input)

    if not gexf_path.is_file():
        print(f"Error: {gexf_path} not found.", file=sys.stderr)
        sys.exit(1)

    # Load graph – ensure undirected
    G = nx.read_gexf(gexf_path)
    if G.is_directed():
        G = G.to_undirected()

    # Determine output location
    graph_name = gexf_path.stem  # e.g., "knn_graph"
    output_root = Path(args.output_base) / graph_name
    metrics_dir = output_root / "Metrics"
    figures_dir = output_root / "Figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_file = metrics_dir / f"{graph_name}_metrics.txt"
    save_metrics_txt(G, metrics_file)

    # Save figure
    figure_file = figures_dir / f"{graph_name}_metrics.png"
    plot_graph_figure(G, figure_file)


if __name__ == "__main__":
    main()