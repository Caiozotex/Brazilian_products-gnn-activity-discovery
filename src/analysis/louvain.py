"""
Louvain community detection — implemented from scratch.

Algorithm (Blondel et al., 2008):
  Repeat until stable:
    Phase 1 — Local moving:
      For each node i (random order):
        - Temporarily remove i from its community
        - Compute the modularity gain DeltaQ of placing i in each
          neighbor community
        - Move i to the community with the highest positive DeltaQ;
          if none, keep i in its (now empty) community
      Repeat sweeps until no node moves.

    Phase 2 — Graph compression:
      Collapse each community into a single super-node.
      Internal edges become self-loops; inter-community edges are summed.
      Repeat Phase 1 on the compressed graph.

Modularity gain formula (remove i from current community, try community C):
  DeltaQ(i -> C) = k_{i,C}/m  -  sigma_tot[C] * k[i] / (2 * m^2)

  where:
    k_{i,C}     = sum of edge weights from i to nodes currently in C
    sigma_tot[C]= sum of degrees of nodes in C (after removing i)
    k[i]        = degree (sum of incident edge weights) of node i
    m           = total edge weight (each undirected edge counted once)
"""

import random
import numpy as np
from collections import defaultdict


# ─────────────────────────────────────────────────────────────
# Phase 1: one full sweep over all nodes
# ─────────────────────────────────────────────────────────────

def _phase1_sweep(adj, k, m, comm, sigma_tot, sigma_in, node_order):
    """
    One sweep over all nodes. Mutates comm, sigma_tot, sigma_in in place.
    Returns True if at least one node changed community.
    """
    improved = False

    for i in node_order:
        c_i = comm[i]

        # Sum of weights from i to its current community (neighbours only, no self)
        k_i_ci = 0.0
        for j, w in adj[i].items():
            if j != i and comm[j] == c_i:
                k_i_ci += w

        # Remove i from c_i
        sigma_tot[c_i] -= k[i]
        sigma_in[c_i]  -= 2.0 * k_i_ci
        comm[i]         = -1          # mark as unassigned

        # Aggregate k_{i→C} for each distinct neighbour community
        neigh_comms = defaultdict(float)
        for j, w in adj[i].items():
            if j == i:
                continue
            cj = comm[j]
            if cj >= 0:
                neigh_comms[cj] += w

        # Evaluate DeltaQ for each candidate community
        best_dQ = 0.0
        best_c  = c_i          # default: stay in (now empty) old community

        for c, k_ic in neigh_comms.items():
            dQ = k_ic / m - sigma_tot[c] * k[i] / (2.0 * m * m)
            if dQ > best_dQ:
                best_dQ = dQ
                best_c  = c

        # Place i in best community
        comm[i]          = best_c
        k_i_best         = neigh_comms.get(best_c, 0.0)
        sigma_tot[best_c] += k[i]
        sigma_in[best_c]  += 2.0 * k_i_best

        if best_c != c_i:
            improved = True

    return improved


# ─────────────────────────────────────────────────────────────
# Phase 2: graph compression
# ─────────────────────────────────────────────────────────────

def _phase2_compress(adj, comm, N):
    """
    Compress communities into super-nodes.

    Returns:
      new_adj      : dict-of-dicts adjacency for compressed graph
      N_new        : number of super-nodes
      comm_to_new  : old community id -> new node index
      node_to_orig : new node index -> frozenset of original nodes in super-node
    """
    unique_comms = sorted(set(comm))
    comm_to_new  = {c: idx for idx, c in enumerate(unique_comms)}
    N_new        = len(unique_comms)

    # Accumulate edges in compressed graph
    new_adj_raw = defaultdict(lambda: defaultdict(float))
    for i in range(N):
        ci = comm_to_new[comm[i]]
        for j, w in adj[i].items():
            cj = comm_to_new[comm[j]]
            new_adj_raw[ci][cj] += w

    new_adj = {i: dict(new_adj_raw[i]) for i in range(N_new)}
    return new_adj, N_new, comm_to_new


# ─────────────────────────────────────────────────────────────
# Full Louvain
# ─────────────────────────────────────────────────────────────

def louvain(N, rows_i, rows_j, edge_weights, max_phases=20, max_sweeps=100,
            seed=42, verbose=True):
    """
    Full two-phase Louvain community detection.

    Parameters
    ----------
    N            : number of nodes
    rows_i, rows_j : 1-D arrays of edge endpoints (upper triangle)
    edge_weights : 1-D array of edge weights (same length as rows_i)
    max_phases   : maximum number of Phase-1 + Phase-2 iterations
    max_sweeps   : maximum sweeps per Phase-1 stage
    seed         : random seed for node ordering in Phase 1

    Returns
    -------
    labels : np.ndarray of shape (N,) with community id for each original node
    """
    rng = random.Random(seed)
    m   = float(np.sum(edge_weights))   # total edge weight

    if m == 0:
        return np.arange(N, dtype=int)

    # Build initial adjacency (symmetric, including self-loops after compression)
    def build_adj(n, ri, rj, rw):
        adj = {i: {} for i in range(n)}
        for i, j, w in zip(ri, rj, rw):
            adj[i][j] = adj[i].get(j, 0.0) + w
            if i != j:
                adj[j][i] = adj[j].get(i, 0.0) + w
        return adj

    adj = build_adj(N,
                    rows_i.tolist(),
                    rows_j.tolist(),
                    edge_weights.tolist())

    # Mapping: current node index -> set of original node indices
    node_to_orig = {i: {i} for i in range(N)}

    # Current graph size
    N_curr = N
    final_labels = np.arange(N, dtype=int)  # will be updated each phase

    for phase in range(max_phases):

        # ── Phase 1 ───────────────────────────────────────────
        # Initial partition: each current node in its own community
        comm      = list(range(N_curr))
        k         = {i: sum(adj[i].values()) for i in range(N_curr)}
        sigma_tot = {i: k[i]                 for i in range(N_curr)}
        sigma_in  = {i: adj[i].get(i, 0.0) * 2.0 for i in range(N_curr)}

        node_order = list(range(N_curr))
        rng.shuffle(node_order)

        for sweep in range(max_sweeps):
            moved = _phase1_sweep(adj, k, m, comm, sigma_tot, sigma_in, node_order)
            rng.shuffle(node_order)
            if not moved:
                if verbose:
                    n_comms = len(set(comm))
                    print(f"  Phase {phase+1}: {N_curr} nodes -> "
                          f"{n_comms} communities ({sweep+1} sweeps)")
                break

        # Update original-node labels through hierarchy
        unique_comms = sorted(set(comm))
        comm_to_idx  = {c: idx for idx, c in enumerate(unique_comms)}

        new_node_to_orig = {}
        for c in unique_comms:
            idx      = comm_to_idx[c]
            members  = set()
            for curr in range(N_curr):
                if comm[curr] == c:
                    members |= node_to_orig[curr]
            new_node_to_orig[idx] = members

        node_to_orig = new_node_to_orig

        # Write current community assignment back to original nodes
        for new_idx, orig_nodes in node_to_orig.items():
            for orig in orig_nodes:
                final_labels[orig] = new_idx

        # Check convergence: no merges happened
        if len(unique_comms) == N_curr:
            if verbose:
                print(f"  Converged at phase {phase+1} "
                      f"(no merges, {N_curr} communities).")
            break

        # ── Phase 2 ───────────────────────────────────────────
        new_adj, N_new, _ = _phase2_compress(adj, comm, N_curr)
        adj    = new_adj
        N_curr = N_new

        if N_curr == 1:
            if verbose:
                print("  All nodes merged into a single community.")
            break

    # Renumber final labels 0..K-1
    unique_final = sorted(set(final_labels.tolist()))
    remap        = {c: i for i, c in enumerate(unique_final)}
    final_labels = np.array([remap[c] for c in final_labels.tolist()], dtype=int)

    if verbose:
        print(f"  Final communities: {len(unique_final)}")

    return final_labels


# ─────────────────────────────────────────────────────────────
# Modularity score (for verification)
# ─────────────────────────────────────────────────────────────

def modularity(N, rows_i, rows_j, edge_weights, labels):
    """Compute modularity Q for a given partition."""
    m = float(np.sum(edge_weights))
    if m == 0:
        return 0.0

    k = np.zeros(N)
    for i, j, w in zip(rows_i, rows_j, edge_weights):
        k[i] += w
        k[j] += w

    Q = 0.0
    for i, j, w in zip(rows_i, rows_j, edge_weights):
        if labels[i] == labels[j]:
            Q += w - k[i] * k[j] / (2 * m)

    return Q / m
