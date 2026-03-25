# BrNPDB-GNN: Transfer Learning for Brazilian Natural Products

## Setup
```bash
conda env create -f environment.yml
conda activate brnpdb-gnn
pip install -e .
```

## Run
```bash
python src/train/pretrain.py --config configs/pretrain.yaml
python src/train/finetune.py --config configs/finetune.yaml
jupyter lab notebooks/
```
