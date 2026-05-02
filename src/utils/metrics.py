"""
Graph metrics and visualisation for GEXF molecular graphs.

Reads all .gefx files from a given directory, computes basic graph
statistics, saves a text summary in results/<folder>/Metrics/,
and saves a figure in results/<folder>/Figures/.
"""

import os
import sys
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute graph metrics and plots for GEXF graphs."
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to the folder containing .gefx files.",
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="results",
        help="Base directory for results (default: results).",
    )
    return parser.parse_args()


def save_metrics_txt(G, filepath):
    """Write the basic graph metrics to a text file."""
    n_nodes = nx.number_of_nodes(G)
    n_edges = nx.number_of_edges(G)
    degrees = dict(G.degree())
    avg_degree = np.mean(list(degrees.values()))
    n_components = nx.number_connected_components(G)
    
    # collect component sizes sorted descending
    comp_sizes = sorted(
        [len(c) for c in nx.connected_components(G)], reverse=True
    )

    with open(filepath, "w") as f:
        f.write(f"Number of vertices: {n_nodes}\n")
        f.write(f"Number of edges: {n_edges}\n")
        f.write(f"Average degree: {avg_degree:.2f}\n")
        f.write(f"Number of connected components: {n_components}\n")
        f.write(f"Component sizes (sorted): {comp_sizes}\n")


def plot_graph_figure(G, figure_path):
    """
    Generate a figure with two subplots:
    - Degree distribution (log-log)
    - Component size distribution (if more than one component)
    """
    # Determine number of components and component sizes
    components = list(nx.connected_components(G))
    n_components = len(components)
    comp_sizes = [len(c) for c in components]

    # Create figure with one or two subplots
    if n_components > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))
        ax2 = None

    # ---- Degree distribution ----
    # Use networkx built-in histogram: index i = degree i, value = count
    degree_hist = nx.degree_histogram(G)
    # Remove degrees with zero count (helps log-log plotting)
    degrees = np.nonzero(degree_hist)[0]
    counts = np.array(degree_hist)[degrees]

    ax1.loglog(degrees, counts, "o", markersize=5, alpha=0.7)
    ax1.set_xlabel("Degree (log)")
    ax1.set_ylabel("Frequency (log)")
    ax1.set_title("Degree Distribution (log-log)")
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # ---- Component size distribution ----
    if ax2 is not None:
        # Count how many components have each distinct size
        sizes, freq = np.unique(comp_sizes, return_counts=True)
        ax2.bar(sizes, freq, width=1.0, edgecolor="black")
        ax2.set_xlabel("Component size (number of vertices)")
        ax2.set_ylabel("Number of components")
        ax2.set_title("Component Size Distribution")
        ax2.grid(axis="y", alpha=0.5)

    elif n_components == 1:
        # If only one component, optionally note that in the single subplot
        ax1.text(
            0.5,
            -0.15,
            "Only one connected component",
            transform=ax1.transAxes,
            ha="center",
            fontsize=12,
        )

    plt.tight_layout()
    plt.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_gexf_file(gexf_path, metrics_dir, figures_dir):
    """Process a single GEXF file: compute metrics, save txt and figure."""
    # Load the graph
    G = nx.read_gexf(gexf_path)
    # GEXF often loads as directed; convert to undirected if necessary
    if G.is_directed():
        G = G.to_undirected()

    # Use the file stem as ID (e.g. "0" for "0.gefx")
    stem = Path(gexf_path).stem

    # Save metrics
    metrics_file = metrics_dir / f"{stem}.txt"
    save_metrics_txt(G, metrics_file)

    # Save figure
    figure_file = figures_dir / f"{stem}_figure1.png"
    plot_graph_figure(G, figure_file)

    print(f"Processed {gexf_path}")


def main():
    args = parse_args()
    input_dir = Path(args.input_folder)

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    # Determine output folder name from the input folder
    output_folder_name = input_dir.name  # e.g., "graphs_clintox"
    output_base = Path(args.output_base)
    output_root = output_base / output_folder_name
    metrics_dir = output_root / "Metrics"
    figures_dir = output_root / "Figures"

    # Create output directories
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Find all .gexf files in the input directory
    gefx_files = sorted(input_dir.glob("*.gexf"))

    if not gefx_files:
        print(f"No .gexf files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    for gfile in gefx_files:
        process_gexf_file(gfile, metrics_dir, figures_dir)


if __name__ == "__main__":
    main()