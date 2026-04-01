# Download HIV,Clintox,Tox21,MUV,BBBP datasets

import os
import urllib.request
import gzip
import shutil

# ---------------------------------
# Folder where datasets will be saved
# ---------------------------------
DATA_DIR = "data/external"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------
# Dataset URLs + final filenames
# ---------------------------------
datasets = {
    "BBBP": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "filename": "BBBP.csv",
        "compressed": False
    },
    "HIV": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
        "filename": "HIV.csv",
        "compressed": False
    },
    "clintox": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
        "filename": "clintox.csv",
        "compressed": True
    },
    "tox21": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        "filename": "tox21.csv",
        "compressed": True
    },
    "muv": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz",
        "filename": "muv.csv",
        "compressed": True
    }
}

# ---------------------------------
# Download function
# ---------------------------------
def download_file(url, output_path):
    print(f"Downloading {output_path}...")
    urllib.request.urlretrieve(url, output_path)
    print("Done.")

# ---------------------------------
# Main loop
# ---------------------------------
for name, info in datasets.items():

    final_path = os.path.join(DATA_DIR, info["filename"])

    # Skip if already downloaded
    if os.path.exists(final_path):
        print(f"{info['filename']} already exists. Skipping.")
        continue

    if info["compressed"]:
        temp_gz = os.path.join(DATA_DIR, name + ".csv.gz")

        download_file(info["url"], temp_gz)

        print(f"Extracting {info['filename']}...")
        with gzip.open(temp_gz, "rb") as f_in:
            with open(final_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(temp_gz)
        print("Extraction finished.\n")

    else:
        download_file(info["url"], final_path)
        print()

print("All benchmark datasets are ready in data/external/")