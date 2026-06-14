import torch

def build_normalized_adj(edge_index, num_nodes, edge_weight=None):
    """
    Returns sparse normalized adjacency S = D^{-1/2} A D^{-1/2}
    """
    row, col = edge_index

    if edge_weight is None:
        edge_weight = torch.ones(row.size(0), device=row.device)

    # Degree
    deg = torch.zeros(num_nodes, device=row.device)
    deg.scatter_add_(0, row, edge_weight)

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    # Normalize weights
    norm_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    S = torch.sparse_coo_tensor(
        edge_index,
        norm_weight,
        (num_nodes, num_nodes)
    )

    return S.coalesce()

def build_label_matrix(hiv_active):
    """
    hiv_active: tensor [N] with values {-1, 0, 1}
    Returns:
        Y: [N, 2]
        labeled_mask: [N]
    """
    N = hiv_active.size(0)

    Y = torch.zeros((N, 2), dtype=torch.float, device=hiv_active.device)

    labeled_mask = hiv_active != -1

    active_weight = 39684 / 1443   

    # inactive → [1, 0]
    Y[(hiv_active == 0), 0] = 1.0

    # active → [0, 1]
    Y[(hiv_active == 1), 1] = active_weight

    return Y, labeled_mask

def label_propagation(
    edge_index,
    num_nodes,
    hiv_active,
    edge_weight=None,
    alpha=0.9,
    max_iter=100,
    tol=1e-6,
    device="cpu"
):
    """
    Returns:
        F: [N, 2] propagated scores
    """

    edge_index = edge_index.to(device)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)

    hiv_active = hiv_active.to(device)

    # Build normalized graph
    S = build_normalized_adj(edge_index, num_nodes, edge_weight)

    # Labels
    Y, labeled_mask = build_label_matrix(hiv_active)

    # Initialize
    F = Y.clone()

    for it in range(max_iter):
        F_old = F.clone()

        # Propagation step
        F = alpha * torch.sparse.mm(S, F) + (1 - alpha) * Y

        # ✅ Normalize row-wise (probabilities)
        F = F / (F.sum(dim=1, keepdim=True) + 1e-8)

        # 🔒 Clamp labeled nodes (anchor)
        F[labeled_mask] = Y[labeled_mask]

        # Convergence check
        delta = torch.norm(F - F_old, p=2)
        if delta < tol:
            print(f"Converged at iter {it}, delta={delta:.2e}")
            break

    return F

def get_unlabeled_predictions(F, hiv_active):
    """
    Returns probability of being ACTIVE for unlabeled nodes
    """
    probs_active = F[:, 1] / (F.sum(dim=1) + 1e-8)

    unlabeled_mask = hiv_active == -1

    return probs_active[unlabeled_mask]

def compute_edge_homophily(edge_index, hiv_active):
    src, dst = edge_index

    mask = (hiv_active[src] != -1) & (hiv_active[dst] != -1)

    src = src[mask]
    dst = dst[mask]

    same = (hiv_active[src] == hiv_active[dst]).float()

    homophily = same.mean().item()

    print(f"Edge homophily: {homophily:.4f}")
    return homophily