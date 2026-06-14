# Brazilian Products GNN Activity Discovery

Graph Neural Network (GNN) pipeline for discovering potential HIV-active compounds from the Brazilian Natural Products Database (BrNPDB).

This project combines:

1. GINE-based molecular representation learning
2. HIV activity prediction using a supervised classifier
3. Similarity graph construction using molecular embeddings
4. Label Propagation for semi-supervised inference
5. Label Propagation Algorithm (LPA) community detection
6. Consensus ranking of candidate compounds

The goal is to identify Brazilian natural products with potential anti-HIV activity.

---

# Repository Structure

```text
Brazilian_products-gnn-activity-discovery/

├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── models/
│   └── checkpoints/
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── similiarity_graph_hiv/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── train/
│   ├── utils/
│   └── evaluate/
│
├── environment.yml
├── setup.py
└── README.md
```

---

# Installation

## 1. Clone repository

```bash
git clone https://github.com/<username>/Brazilian_products-gnn-activity-discovery.git

cd Brazilian_products-gnn-activity-discovery
```

---

## 2. Create Conda Environment (Recommended)

```bash
conda env create -f environment.yml
```

Activate environment:

```bash
conda activate nubbe-env
```

Verify installation:

```bash
python --version
```

# OR pip
pip install -r requirements.txt

---

# Data Preparation

The project expects molecular graph files stored as PyTorch Geometric `Data` objects.

Expected directories:

```text
data/
└── processed/
    ├── graphs_hiv/
    └── graphs_brnpdb_antiviral/
```

Each graph should contain:

```python
data.x
data.edge_index
data.edge_attr
```

For HIV molecules:

```python
data.hiv_active
```

For BrNPDB molecules:

```python
data.brnpdb_id
data.common_name
```

---

# Reproducing Results

The HIV screening pipeline can be reproduced with the following commands.

---

# Step 1 — HIV Screening of BrNPDB

Evaluate the pretrained classifier on HIV molecules.

Example:

```bash
python -m src.train.evaluate_hiv_model
```

Metrics:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score


# Step 2 — Label Propagation

Run graph-based label propagation on the similarity graph.

Example:

```bash
python -m src.train.evaluate_label_prop
```

---

# Step 3 — Community Detection (LPA)

Detect communities in the similarity graph.

Example:

```bash
python -m src.train.evaluate_lpa
```

---

# Step 4 — Consensus Screening

Combine:

- HIV classifier predictions
- Label propagation predictions
- Community assignments

Example:

```bash
python -m src.utils.brnpdb_consensus_screening
```

---

## Pipeline Summary

```text
HIV Dataset
      │
      ▼
Train GINE Encoder
      │
      ▼
Generate Molecular Embeddings
      │
      ▼
Build Similarity Graph
      │
      ├──────────────► HIV Classifier Screening
      │
      ├──────────────► Label Propagation
      │
      └──────────────► Community Detection (LPA)
                             │
                             ▼
                  Consensus Ranking
```

---


# Main Result Files

| File | Description |
|--------|--------|
| results/tables/brnpdb_hiv_screening.csv | HIV classifier predictions |
| results/tables/label_propagation_screening.csv| Label propagation predictions |
| results/tables/lpa_community_members.csv | Community assignments |
| results/tables/brnpdb_consensus_screening.csv | Combined ranking |
| results/similiarity_graph_hiv/hiv_knn_graph/graph_hiv_communities.gexf | Graph visualization with communities|

---

# Methodology

1. Convert molecules into graphs.
2. Train a GINE encoder on HIV activity data.
3. Generate molecular embeddings.
4. Build a similarity graph using k-NN.
5. Predict HIV activity using:
   - Supervised classifier
   - Label Propagation
   - Community analysis
6. Combine predictions into a consensus ranking.

---

# Citation

If you use this repository, please cite:

```text
Predição de Atividades Biológicas de Produtos Naturais Brasileiros usando Redes Neurais de Grafos (GNN)

Universidade Estadual de Campinas (UNICAMP)
MC859 - Projeto em Teoria da Computação
2026
```

---

# License

MIT License