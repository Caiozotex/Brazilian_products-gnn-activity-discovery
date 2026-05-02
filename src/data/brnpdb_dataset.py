import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Union, Dict, Any
from tqdm import tqdm

from torch_geometric.data import Data, Batch

from src.utils.rdkit_utils import smiles_to_mol, mol_to_graph


class BRNPDBDataset(Dataset):
    """Dataset for BRNPDB compounds using lazy loading of PyG graphs."""

    def __init__(
        self,
        csv_path: str,
        processed_dir: str = "data/processed",
        force_reprocess: bool = False,
        return_properties: bool = True,
    ):
        self.csv_path = csv_path
        self.processed_dir = processed_dir
        self.force_reprocess = force_reprocess
        self.return_properties = return_properties

        # Load CSV
        self.df = pd.read_csv(csv_path)

        # Unique ID (prefer inchikey)
        if "inchikey" in self.df.columns:
            self.df["_id"] = self.df["inchikey"].astype(str)
        else:
            self.df["_id"] = self.df.index.astype(str)

        os.makedirs(self.processed_dir, exist_ok=True)

        # Preprocess only (do NOT store graphs in memory)
        self._process_all()

    # -------------------------
    # Preprocessing
    # -------------------------
    def _process_all(self):
        """Generate and save graphs (.pt) if needed."""
        for _, row in tqdm(
            self.df.iterrows(),
            total=len(self.df),
            desc="Processing compounds",
        ):
            comp_id = row["_id"]
            smiles = row.get("smiles", None)

            if not isinstance(smiles, str) or len(smiles) == 0:
                print(f"Warning: No valid SMILES for {comp_id}, skipping.")
                continue

            save_path = os.path.join(self.processed_dir, f"{comp_id}.pt")

            # Skip if already exists
            if not self.force_reprocess and os.path.exists(save_path):
                continue

            # Convert SMILES → Mol
            mol = smiles_to_mol(smiles)
            if mol is None:
                print(f"Warning: Invalid SMILES for {comp_id}")
                continue

            # Convert Mol → Graph
            data = mol_to_graph(mol)
            if data is None:
                print(f"Warning: Failed to convert mol for {comp_id}")
                continue

            # -------------------------
            # Add molecular properties
            # -------------------------
            for prop in [
                "mw", "logp", "tpsa", "hba", "hbd",
                "nrotb", "volume", "monomass"
            ]:
                if prop in row.index and pd.notna(row[prop]):
                    value = float(row[prop])
                    setattr(data, prop, torch.tensor(value, dtype=torch.float))

            # -------------------------
            # Minimal label (raw string)
            # -------------------------
            if "bioactivities" in row and isinstance(row["bioactivities"], str):
                data.y = row["bioactivities"]

            # Save graph
            torch.save(data, save_path)

    # -------------------------
    # Dataset interface
    # -------------------------
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> Union[Data, tuple]:
        row = self.df.iloc[idx]
        comp_id = row["_id"]
        path = os.path.join(self.processed_dir, f"{comp_id}.pt")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Graph file not found for {comp_id}")

        data = torch.load(path)

        if self.return_properties:
            prop_dict: Dict[str, Any] = {
                k: row[k]
                for k in [
                    "mw", "logp", "tpsa", "hba", "hbd",
                    "nrotb", "volume", "monomass"
                ]
                if k in row.index and pd.notna(row[k])
            }
            return data, prop_dict

        return data

    # -------------------------
    # Collate function
    # -------------------------
    @staticmethod
    def collate_fn(batch):
        if isinstance(batch[0], tuple):
            graphs, props = zip(*batch)
            return Batch.from_data_list(graphs), list(props)
        else:
            return Batch.from_data_list(batch)

class ClintoxDataset(Dataset):
    """
    Dataset for clintox.csv with columns: smiles, FDA_APPROVED, CT_TOX
    - FDA_APPROVED: 1 = approved, 0 = not approved
    - CT_TOX: 1 = toxic (failed trial), 0 = non-toxic
    """

    def __init__(self, csv_path: str, processed_dir: str = "data/processed_clintox",
                 force_reprocess: bool = False, return_labels: bool = True):
        self.csv_path = csv_path
        self.processed_dir = processed_dir
        self.force_reprocess = force_reprocess
        self.return_labels = return_labels

        self.df = pd.read_csv(csv_path)
        # Use row index as ID (or if there is a molecule id column, use that)
        self.df['_id'] = self.df.index.astype(str)

        os.makedirs(self.processed_dir, exist_ok=True)
        self.graphs = []
        self.labels = []   # list of [fda_approved, ct_tox]
        self._process_all()

    def _process_all(self):
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing Clintox"):
            comp_id = row['_id']
            smiles = row.get('smiles', None)
            if not isinstance(smiles, str) or len(smiles) == 0:
                print(f"Warning: No SMILES for {comp_id}, skipping.")
                continue

            save_path = os.path.join(self.processed_dir, f"{comp_id}.pt")
            if not self.force_reprocess and os.path.exists(save_path):
                data = torch.load(save_path)
            else:
                mol = smiles_to_mol(smiles)
                if mol is None:
                    print(f"Warning: Invalid SMILES for {comp_id}: {smiles}")
                    continue
                data = mol_to_graph(mol)
                if data is None:
                    continue
                # Add labels as graph attributes
                fda = float(row['FDA_APPROVED'])
                ctox = float(row['CT_TOX'])
                data.fda_approved = torch.tensor(fda, dtype=torch.float)
                data.ct_tox = torch.tensor(ctox, dtype=torch.float)
                # Also combine into a single label tensor if desired
                data.y = torch.tensor([fda, ctox], dtype=torch.float)
                torch.save(data, save_path)

            self.graphs.append(data)
            if self.return_labels:
                self.labels.append([data.fda_approved.item(), data.ct_tox.item()])

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx]
        if self.return_labels:
            return data, self.labels[idx]
        return data

    @staticmethod
    def collate_fn(batch):
        if isinstance(batch[0], tuple):
            graphs, labels = zip(*batch)
            return Batch.from_data_list(graphs), list(labels)
        return Batch.from_data_list(batch)

class Tox21Dataset(Dataset):
    """
    Dataset for tox21.csv.
    Columns: NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD,
             NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53,
             mol_id, smiles
    Each target column contains binary values (1/0) or NaN (missing).
    We'll treat NaN as -1 (or 0, but -1 is typical for missing in Tox21).
    """

    # List of all 12 task names
    TASKS = [
        'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
        'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
    ]

    def __init__(self, csv_path: str, processed_dir: str = "data/processed_tox21",
                 force_reprocess: bool = False, return_labels: bool = True,
                 nan_value: float = -1.0):
        self.csv_path = csv_path
        self.processed_dir = processed_dir
        self.force_reprocess = force_reprocess
        self.return_labels = return_labels
        self.nan_value = nan_value

        self.df = pd.read_csv(csv_path)
        # Use mol_id if present, else index
        if 'mol_id' in self.df.columns:
            self.df['_id'] = self.df['mol_id'].astype(str)
        else:
            self.df['_id'] = self.df.index.astype(str)

        os.makedirs(self.processed_dir, exist_ok=True)
        self.graphs = []
        self.labels = []   # list of 12‑element tensors
        self._process_all()

    def _process_all(self):
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing Tox21"):
            comp_id = row['_id']
            smiles = row.get('smiles', None)
            if not isinstance(smiles, str) or len(smiles) == 0:
                print(f"Warning: No SMILES for {comp_id}, skipping.")
                continue

            save_path = os.path.join(self.processed_dir, f"{comp_id}.pt")
            if not self.force_reprocess and os.path.exists(save_path):
                data = torch.load(save_path)
            else:
                mol = smiles_to_mol(smiles)
                if mol is None:
                    print(f"Warning: Invalid SMILES for {comp_id}: {smiles}")
                    continue
                data = mol_to_graph(mol)
                if data is None:
                    continue

                # Extract labels for the 12 tasks, replace NaN with nan_value
                label_values = []
                for task in self.TASKS:
                    val = row[task]
                    if pd.isna(val):
                        label_values.append(self.nan_value)
                    else:
                        label_values.append(float(val))
                data.y = torch.tensor(label_values, dtype=torch.float)
                # Also store each task individually if needed
                for i, task in enumerate(self.TASKS):
                    setattr(data, task.replace('-', '_'), data.y[i])
                torch.save(data, save_path)

            self.graphs.append(data)
            if self.return_labels:
                self.labels.append(data.y.clone())

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx]
        if self.return_labels:
            return data, self.labels[idx]
        return data

    @staticmethod
    def collate_fn(batch):
        if isinstance(batch[0], tuple):
            graphs, labels = zip(*batch)
            return Batch.from_data_list(graphs), list(labels)
        return Batch.from_data_list(batch)
        

# Create the directorys if it doesn't exist
PROCESSED_DIR_CytotoxicAnticancer = "data/processed/graphs_CytotoxicAnticancer"
os.makedirs(PROCESSED_DIR_CytotoxicAnticancer, exist_ok=True)

PROCESSED_DIR_AntioxidantRos = "data/processed/graphs_AntioxidantRos"
os.makedirs(PROCESSED_DIR_AntioxidantRos, exist_ok=True)


PROCESSED_DIR_Clintox = "data/processed/graphs_clintox"
os.makedirs(PROCESSED_DIR_Clintox, exist_ok=True)

PROCESSED_DIR_Tox21 = "data/processed/graphs_tox21"
os.makedirs(PROCESSED_DIR_Tox21, exist_ok=True)


#=================Cytotoxic and Anticancer graph compounds===============
dataset_1 = BRNPDBDataset(
    csv_path="data/processed/nubbe_cytotoxic_anticancer.csv",
    processed_dir="data/processed/graphs_CytotoxicAnticancer",
)

print("Dataset size:", len(dataset_1))

data, props = dataset_1[0]

print(data)
print("Num nodes:", data.num_nodes)
print("Properties:", props)
print("Label:", data.y)


#=================Antioxidant and ROS graph compounds===============
dataset_2 = BRNPDBDataset(
    csv_path="data/processed/nubbe_antioxidant_ros.csv",
    processed_dir="data/processed/graphs_AntioxidantRos",
)

print("Dataset size:", len(dataset_2))

data, props = dataset_2[0]

print(data)
print("Num nodes:", data.num_nodes)
print("Properties:", props)
print("Label:", data.y)

#=================Clintox graph compounds===============
dataset_3 = ClintoxDataset(
    csv_path="data/external/clintox.csv",
    processed_dir="data/processed/graphs_clintox",
)

print("Dataset size:", len(dataset_3))

data, props = dataset_3[0]

print(data)
print("Num nodes:", data.num_nodes)
print("Properties:", props)
print("Label:", data.y)

#=================Tox21 graph compounds===============
dataset_4 = Tox21Dataset(
    csv_path="data/external/tox21.csv",
    processed_dir="data/processed/graphs_tox21",
)

print("Dataset size:", len(dataset_4))

data, props = dataset_4[0]

print(data)
print("Num nodes:", data.num_nodes)
print("Properties:", props)
print("Label:", data.y)