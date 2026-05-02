import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from torch_geometric.utils import degree, to_networkx
import torch_geometric
import networkx as nx
import os

def analyze_graph_metrics(graph_path):
    """
    Analyze graph metrics: connected components, average degree, and degree distribution.

    Args:
        graph_path (str): Path to the PyTorch Geometric graph file (.pt)
    """
    print("Loading graph...")
    # Load the graph
    with torch.serialization.safe_globals([torch_geometric.data.Data]):
        graph_data = torch.load(graph_path, weights_only=False)

    # Convert to NetworkX for easier analysis
    G = to_networkx(graph_data, to_undirected=True)

    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 1. Calculate connected components
    print("\nCalculating connected components...")
    num_components = nx.number_connected_components(G)
    component_sizes = [len(component) for component in nx.connected_components(G)]
    component_sizes_sorted = sorted(component_sizes, reverse=True)

    print(f"Number of connected components: {num_components}")
    print(f"Component sizes (sorted): {component_sizes_sorted}")

    # 2. Calculate average degree
    print("\nCalculating degree statistics...")
    degrees = [d for n, d in G.degree()]
    avg_degree = np.mean(degrees)
    min_degree = np.min(degrees)
    max_degree = np.max(degrees)
    median_degree = np.median(degrees)

    print(f"Number of vertices: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average degree: {avg_degree:.2f}")
    print(f"Min degree: {min_degree}")
    print(f"Max degree: {max_degree}")
    print(f"Median degree: {median_degree}")

    # 3. Create degree distribution histogram
    print("\nCreating degree distribution histogram...")
    try:
        print(f"Degrees array length: {len(degrees)}")
        print(f"Min degree: {min_degree}, Max degree: {max_degree}")
        print(f"Sample degrees: {degrees[:10]}")

        plt.figure(figsize=(12, 8))

        # Create histogram with simpler bins
        bins = 31  # 0 to 30
        plt.hist(degrees, bins=bins, alpha=0.7, edgecolor='black', align='left')

        plt.xlabel('Degree (Number of Connections)', fontsize=12)
        plt.ylabel('Frequency (Number of Vertices)', fontsize=12)
        plt.title('Degree Distribution of Similarity Graph\n(Threshold = 0.95)', fontsize=14, pad=20)

        # Add grid
        plt.grid(True, alpha=0.3)

        # Add statistics text box
        stats_text = '.2f' + '.2f' + '.2f' + '.2f' + '.2f' + '.2f'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Save the plot
        output_dir = os.path.join(os.path.dirname(graph_path), 'Figures')
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, 'knn_graph_metrics.png')
        plt.savefig(plot_path, dpi=150)
        print(f"Degree distribution plot saved to: {plot_path}")

        # Don't close the plot to avoid issues
        # plt.close()
    except Exception as e:
        print(f"Error creating histogram: {e}")
        import traceback
        traceback.print_exc()
        print("Skipping histogram creation...")

    # Save metrics to text file
    metrics_dir = os.path.join(os.path.dirname(graph_path), 'Metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, 'knn_graph_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Number of vertices: {G.number_of_nodes()}\n")
        f.write(f"Number of edges: {G.number_of_edges()}\n")
        f.write(f"Average degree: {avg_degree:.2f}\n")
        f.write(f"Number of connected components: {num_components}\n")
        f.write(f"Component sizes (sorted): {component_sizes_sorted}\n")
        f.write(f"\nDegree Statistics:\n")
        f.write(f"Min degree: {min_degree}\n")
        f.write(f"Max degree: {max_degree}\n")
        f.write(f"Median degree: {median_degree}\n")
        f.write(f"Standard deviation: {np.std(degrees):.2f}\n")

    print(f"Metrics saved to: {metrics_path}")

    # Save summary to separate file
    summary_path = os.path.join(os.path.dirname(graph_path), 'graph_analysis.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Number of vertices: {G.number_of_nodes()}\n")
        f.write(f"Number of edges: {G.number_of_edges()}\n")
        f.write(f"Average degree: {avg_degree:.2f}\n")
        f.write(f"Number of components: {num_components}\n")
    print(f"Summary saved to: {summary_path}")

    # Don't show the plot to avoid blocking
    # plt.show()

    print("Returning from function...")
    return {
        'num_vertices': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'avg_degree': avg_degree,
        'min_degree': min_degree,
        'max_degree': max_degree,
        'median_degree': median_degree,
        'num_components': num_components,
        'component_sizes': component_sizes_sorted,
        'degrees': degrees
    }

if __name__ == "__main__":
    # Path to the graph file
    graph_path = "results/similiarity_graph/undirected_knn_graph.pt"

    if not os.path.exists(graph_path):
        print(f"Error: Graph file not found at {graph_path}")
        print("Please run the evaluation script first to generate the graph.")
        exit(1)

    # Analyze the graph
    metrics = analyze_graph_metrics(graph_path)

    if metrics is None:
        print("Error: Failed to analyze graph metrics.")
        exit(1)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Graph: {metrics['num_vertices']} vertices, {metrics['num_edges']} edges")
    print(f"Average degree: {metrics['avg_degree']:.2f}")
    print(f"Min degree: {metrics['min_degree']}")
    print(f"Max degree: {metrics['max_degree']}")
    print(f"Median degree: {metrics['median_degree']}")
    print(f"Number of connected components: {metrics['num_components']}")
    print(f"Component sizes (sorted): {metrics['component_sizes']}")

    # Files are already saved by the analyze_graph_metrics function
    print("Analysis complete! Files saved in results/similiarity_graph/")
    print(f"Connected components: {metrics['num_components']}")
    print("Files generated:")
    print("  - results/similiarity_graph/Metrics/knn_graph_metrics.txt")
    print("  - results/similiarity_graph/Figures/knn_graph_metrics.png")