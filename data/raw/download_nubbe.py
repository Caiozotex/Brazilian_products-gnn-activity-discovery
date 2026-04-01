# Download NuBBE[KG]

import os
import urllib.request
from rdflib import Graph
import pandas as pd


# ---------------------------
# Paths
# ---------------------------
DATA_EXTERNAL = "data/external"
DATA_RAW = "data/raw"

os.makedirs(DATA_EXTERNAL, exist_ok=True)
os.makedirs(DATA_RAW, exist_ok=True)


# ---------------------------
# Download RDF
# ---------------------------
url = "https://nubbekg.aksw.org/downloads/dumps/nubbe-knowledge-graph-v2.0.0.rc1.ttl"
file_path = os.path.join(DATA_EXTERNAL, "nubbe-knowledge-graph-v2.0.0.rc1.ttl")

if not os.path.exists(file_path):
    print("Downloading NuBBE KG...")
    urllib.request.urlretrieve(url, file_path)
    print("Download finished.")
else:
    print("File already exists.")


# ---------------------------
# Load graph
# ---------------------------
g = Graph()
g.parse(file_path, format="turtle")

print("Triples loaded:", len(g))


# ---------------------------
# Query 1: compound + descriptors
# ---------------------------
query_1 = """
SELECT
?compound ?name ?smiles
?mw ?formula ?volume ?monomass ?nrotb ?tpsa
?bioactivity
WHERE {

?compound a <http://nubbekg.aksw.org/ontology#Compound> .

OPTIONAL { ?compound <http://nubbekg.aksw.org/ontology#commonName> ?name . }
OPTIONAL { ?compound <http://nubbekg.aksw.org/ontology#smiles> ?smiles . }

OPTIONAL {
    ?compound <http://nubbekg.aksw.org/ontology#hasDescriptors> ?d .
    ?d a <http://nubbekg.aksw.org/ontology#MolecularDescriptors> .

    OPTIONAL { ?d <http://nubbekg.aksw.org/ontology#molecularWeight> ?mw . }
    OPTIONAL { ?d <http://nubbekg.aksw.org/ontology#molecularFormula> ?formula . }
    OPTIONAL { ?d <http://nubbekg.aksw.org/ontology#molecularVolume> ?volume . }
    OPTIONAL { ?d <http://nubbekg.aksw.org/ontology#monoisotopicMass> ?monomass . }
    OPTIONAL { ?d <http://nubbekg.aksw.org/ontology#nrotb> ?nrotb . }
}

OPTIONAL {
    ?compound <http://nubbekg.aksw.org/ontology#hasDescriptors> ?topo .
    ?topo a <http://nubbekg.aksw.org/ontology#TopologicalDescriptors> .
    OPTIONAL { ?topo <http://nubbekg.aksw.org/ontology#tpsa> ?tpsa . }
}

OPTIONAL {
    ?bio a <http://nubbekg.aksw.org/ontology#Bioactivity> .
    ?bio <http://www.w3.org/2000/01/rdf-schema#label> ?bioactivity .
}

}
"""

results_1 = g.query(query_1)

rows_1 = []
for i, row in enumerate(results_1):
    rows_1.append(row)

    if i % 100 == 0:
        print(f"{i} rows processed...")

df_1 = pd.DataFrame(rows_1)

df_1.columns = [str(v) for v in g.query(query_1).vars]

df_1.to_csv(os.path.join(DATA_RAW, "nubbe_dataset_1.csv"), index=False)


# ---------------------------
# Query 2: identifiers + descriptors
# ---------------------------
query_2 = """
SELECT
?label ?iupac ?inchi ?inchikey
?hba ?hbd ?logp ?lipinski
WHERE {

?mol a <http://nubbekg.aksw.org/ontology#UniqueIdentifiers> ;
     <http://www.w3.org/2000/01/rdf-schema#label> ?label .

OPTIONAL { ?mol <http://nubbekg.aksw.org/ontology#iupacName> ?iupac . }
OPTIONAL { ?mol <http://nubbekg.aksw.org/ontology#inchi> ?inchi . }
OPTIONAL { ?mol <http://nubbekg.aksw.org/ontology#inchikey> ?inchikey . }

OPTIONAL {
    ?ed a <http://nubbekg.aksw.org/ontology#ElectronicDescriptors> ;
        <http://www.w3.org/2000/01/rdf-schema#label> ?label ;
        <http://nubbekg.aksw.org/ontology#hBondAcceptorCount> ?hba ;
        <http://nubbekg.aksw.org/ontology#hBondDonorCount> ?hbd .
}

OPTIONAL {
    ?cd a <http://nubbekg.aksw.org/ontology#ConstitutionalDescriptors> ;
        <http://www.w3.org/2000/01/rdf-schema#label> ?label ;
        <http://nubbekg.aksw.org/ontology#logpCoefficient> ?logp ;
        <http://nubbekg.aksw.org/ontology#lipinskiRuleOf5Failures> ?lipinski .
}

}
"""

results_2 = g.query(query_2)

rows_2 = []
for j, row in enumerate(results_2):
    rows_2.append(row)

    if j % 100 == 0:
        print(f"{j} rows processed...")

df_2 = pd.DataFrame(rows_2)
df_2.columns = [str(v) for v in g.query(query_2).vars]

df_2.to_csv(os.path.join(DATA_RAW, "nubbe_dataset_2.csv"), index=False)


print("Datasets saved in data/raw/")
