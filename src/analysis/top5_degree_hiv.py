import networkx as nx

GEXF_PATH = "results/similiarity_graph_hiv/hiv_knn_graph.gexf"

G = nx.read_gexf(GEXF_PATH)

degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
top5 = degrees[:5]

print("Top 5 vértices por grau — hiv_knn_graph.gexf")
print(f"{'Rank':<6} {'ID':<10} {'Grau'}")
for rank, (node_id, deg) in enumerate(top5, 1):
    print(f"{rank:<6} {node_id:<10} {deg}")
