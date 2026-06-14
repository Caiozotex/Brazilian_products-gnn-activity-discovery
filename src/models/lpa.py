import torch
from collections import defaultdict
import random
from collections import Counter

def build_adj_list(edge_index, num_nodes):
    adj = [[] for _ in range(num_nodes)]
    src, dst = edge_index

    for i in range(src.size(0)):
        u = src[i].item()
        v = dst[i].item()
        adj[u].append(v)

    return adj


def lpa_community_detection(edge_index, num_nodes, max_iter=50, seed=42):
    random.seed(seed)

    adj = build_adj_list(edge_index, num_nodes)

    # Initialize unique labels
    labels = torch.arange(num_nodes)

    for it in range(max_iter):
        prev_labels = labels.clone()

        nodes = list(range(num_nodes))
        random.shuffle(nodes)  # 🔥 semi-synchronous update

        for node in nodes:
            neighbors = adj[node]

            if len(neighbors) == 0:
                continue

            neighbor_labels = [labels[n].item() for n in neighbors]

            counts = Counter(neighbor_labels)
            max_count = max(counts.values())

            # Random tie-breaking
            best_labels = [lab for lab, c in counts.items() if c == max_count]
            labels[node] = random.choice(best_labels)

        # Convergence check
        if torch.equal(labels, prev_labels):
            print(f"LPA converged at iteration {it}")
            break

    return labels

def lpa_weighted(edge_index, edge_weight, num_nodes, max_iter=50, seed=42):
    random.seed(seed)

    adj = [[] for _ in range(num_nodes)]
    src, dst = edge_index

    for i in range(src.size(0)):
        u = src[i].item()
        v = dst[i].item()
        w = edge_weight[i].item()
        adj[u].append((v, w))

    labels = torch.arange(num_nodes)

    for it in range(max_iter):
        prev_labels = labels.clone()

        nodes = list(range(num_nodes))
        random.shuffle(nodes)

        for node in nodes:
            neighbors = adj[node]
            if not neighbors:
                continue

            scores = defaultdict(float)

            for neigh, w in neighbors:
                scores[labels[neigh].item()] += w

            max_score = max(scores.values())
            best_labels = [lab for lab, s in scores.items() if s == max_score]

            labels[node] = random.choice(best_labels)

        if torch.equal(labels, prev_labels):
            print(f"Weighted LPA converged at iteration {it}")
            break

    return labels