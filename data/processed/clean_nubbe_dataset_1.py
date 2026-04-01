import os
import pandas as pd

# --------------------------------------------------
# Paths
# --------------------------------------------------
RAW_PATH = "data/raw/nubbe_dataset_1.csv"
PROCESSED_DIR = "data/processed"
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "nubbe_dataset_1_clean.csv")

# Create processed folder if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
print("Loading dataset...")
df = pd.read_csv(RAW_PATH)

print(f"Original number of rows: {len(df)}")

# --------------------------------------------------
# Aggregate bioactivities (1 compound = 1 row)
# --------------------------------------------------
print("Aggregating bioactivities...")

df_clean = (
    df
    .groupby(
        ["compound", "name", "smiles", "mw", "formula",
         "volume", "monomass", "nrotb", "tpsa"],
        dropna=False
    )["bioactivity"]
    .apply(lambda x: "; ".join(sorted(set(x.dropna().astype(str)))))
    .reset_index()
)

print(f"New number of rows: {len(df_clean)}")

# --------------------------------------------------
# Save cleaned dataset
# --------------------------------------------------
df_clean.to_csv(OUTPUT_PATH, index=False)

print("Clean dataset saved to:")
print(OUTPUT_PATH)