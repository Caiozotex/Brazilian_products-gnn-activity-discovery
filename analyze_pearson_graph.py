import subprocess
import os
from analyze_graph_metrics import analyze_graph_metrics

threshold = 0.9

print(f"Building Pearson correlation graph for threshold {threshold}")
cmd = f"python build_threshold_graph_pearson.py --threshold {threshold}"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
if result.returncode != 0:
    print(f"Error building: {result.stderr}")
    exit(1)
print(result.stdout)

graph_path = f"results/similiarity_graph/undirected_threshold_pearson_{threshold}_graph.pt"
if not os.path.exists(graph_path):
    print(f"Graph not found for {threshold}")
    exit(1)

print(f"Analyzing Pearson graph for threshold {threshold}")
metrics = analyze_graph_metrics(graph_path)
if metrics is None:
    print(f"Failed to analyze for {threshold}")
    exit(1)

# Save summary
os.makedirs('results/similiarity_graph', exist_ok=True)
with open('results/similiarity_graph/pearson_analysis_summary.txt', 'w') as f:
    f.write("Threshold\tVertices\tEdges\tAvg Degree\tComponents\n")
    f.write(f"{threshold}\t{metrics['num_vertices']}\t{metrics['num_edges']}\t{metrics['avg_degree']:.2f}\t{metrics['num_components']}\n")

print("Pearson analysis complete. Summary saved to results/similiarity_graph/pearson_analysis_summary.txt")