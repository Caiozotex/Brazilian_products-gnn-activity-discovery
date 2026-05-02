import subprocess
import os
from analyze_graph_metrics import analyze_graph_metrics

thresholds = [0.6, 0.8, 0.82, 0.83, 0.85, 0.9, 0.95]

print("Building graphs for all thresholds")
cmd = f"python build_threshold_similarity_graph.py --thresholds {' '.join(map(str, thresholds))}"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
if result.returncode != 0:
    print(f"Error building: {result.stderr}")
    exit(1)
print(result.stdout)

results = {}
for thresh in thresholds:
    graph_path = f"results/similiarity_graph/undirected_threshold_{thresh}_graph.pt"
    if not os.path.exists(graph_path):
        print(f"Graph not found for {thresh}")
        continue

    print(f"Analyzing graph for threshold {thresh}")
    metrics = analyze_graph_metrics(graph_path)
    if metrics is None:
        print(f"Failed to analyze for {thresh}")
        continue

    results[thresh] = {
        'num_vertices': metrics['num_vertices'],
        'num_edges': metrics['num_edges'],
        'avg_degree': metrics['avg_degree'],
        'num_components': metrics['num_components']
    }

# Save summary
os.makedirs('results/similiarity_graph', exist_ok=True)
with open('results/similiarity_graph/threshold_analysis_summary.txt', 'w') as f:
    f.write("Threshold\tVertices\tEdges\tAvg Degree\tComponents\n")
    for thresh in sorted(results.keys()):
        m = results[thresh]
        f.write(f"{thresh}\t{m['num_vertices']}\t{m['num_edges']}\t{m['avg_degree']:.2f}\t{m['num_components']}\n")

print("Threshold analysis complete. Summary saved to results/similiarity_graph/threshold_analysis_summary.txt")