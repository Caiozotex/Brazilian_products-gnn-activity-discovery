# Download NuBBE[KG]

import os
import urllib.request
from rdflib import Graph, URIRef
import pandas as pd
from tqdm import tqdm
import time


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


# ------------------------------------------------------------
# 1. Load graph 
# ------------------------------------------------------------
g = Graph()
g.parse(file_path, format="turtle")

print("Triples loaded:", len(g))

# ---------------------------
# Helper: get all compound URIs
# ---------------------------
def get_all_compounds(g):
    query = """
    PREFIX nubbe: <http://nubbekg.aksw.org/ontology#>
    SELECT DISTINCT ?compound WHERE { ?compound a nubbe:Compound . }
    """
    results = g.query(query)
    return [str(row['compound']) for row in results]

# ---------------------------
# Query templates (no aggregation in SPARQL)
# ---------------------------
# Query 1: Properties + bioactivities
properties_template = """
PREFIX nubbe: <http://nubbekg.aksw.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?compound ?name ?smiles ?mw ?formula ?volume ?monomass ?nrotb ?tpsa ?bioactivityLabel
WHERE {
  VALUES ?compound { <__COMPOUND_URI__> }
  ?compound a nubbe:Compound .
  OPTIONAL { ?compound nubbe:commonName ?name . }
  OPTIONAL { ?compound nubbe:smiles ?smiles . }
  OPTIONAL {
    ?compound nubbe:hasDescriptors ?d .
    OPTIONAL { ?d nubbe:molecularWeight ?mw . }
    OPTIONAL { ?d nubbe:molecularFormula ?formula . }
    OPTIONAL { ?d nubbe:molecularVolume ?volume . }
    OPTIONAL { ?d nubbe:monoisotopicMass ?monomass . }
    OPTIONAL { ?d nubbe:nrotb ?nrotb . }
  }
  OPTIONAL {
    ?compound nubbe:hasDescriptors ?topo .
    OPTIONAL { ?topo nubbe:tpsa ?tpsa . }
  }
  OPTIONAL {
    ?analysis nubbe:discovered ?compound .
    ?analysis nubbe:hasBioAssay ?assay .
    ?assay nubbe:bioactivity ?bioactivity .
    ?bioactivity rdfs:label ?bioactivityLabel .
  }
}
"""

# Query 2: Location (species, state, city)
location_template = """
PREFIX nubbe: <http://nubbekg.aksw.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?compound ?speciesName ?stateName ?cityName
WHERE {
  VALUES ?compound { <__COMPOUND_URI__> }
  ?compound a nubbe:Compound .
  ?analysis nubbe:discovered ?compound .
  ?analysis nubbe:aboutSpecimen ?specimen .
  OPTIONAL { ?specimen nubbe:partOfSpecies ?species . 
             ?species rdfs:label ?speciesName . }
  OPTIONAL {
    ?specimen nubbe:wasDiscoveredIn ?location .
    ?location a nubbe:City .
    ?location nubbe:locatedIn ?state .
    OPTIONAL { ?state rdfs:label ?stateName . }
    OPTIONAL { ?location rdfs:label ?cityName . }
  }
}
"""

# Query 3: Identifiers and extra descriptors (IUPAC, InChI, HBA, HBD, logP, Lipinski)
identifiers_template = """
PREFIX nubbe: <http://nubbekg.aksw.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?compound ?iupac ?inchi ?inchikey ?hba ?hbd ?logp ?lipinski
WHERE {
  VALUES ?compound { <__COMPOUND_URI__> }
  ?compound a nubbe:Compound .

  OPTIONAL {
    ?compound nubbe:isIdentifiedBy ?uid .
    ?uid a nubbe:UniqueIdentifiers .
    OPTIONAL { ?uid nubbe:iupacName ?iupac . }
    OPTIONAL { ?uid nubbe:inchi ?inchi . }
    OPTIONAL { ?uid nubbe:inchikey ?inchikey . }
  }

  OPTIONAL {
    ?compound nubbe:hasDescriptors ?ed .
    ?ed a nubbe:ElectronicDescriptors .
    OPTIONAL { ?ed nubbe:hBondAcceptorCount ?hba . }
    OPTIONAL { ?ed nubbe:hBondDonorCount ?hbd . }
  }

  OPTIONAL {
    ?compound nubbe:hasDescriptors ?cd .
    ?cd a nubbe:ConstitutionalDescriptors .
    OPTIONAL { ?cd nubbe:logpCoefficient ?logp . }
    OPTIONAL { ?cd nubbe:lipinskiRuleOf5Failures ?lipinski . }
  }
}
"""

# ---------------------------
# Query runner
# ---------------------------
def run_query_for_compound(g, compound_uri, query_template):
    query = query_template.replace("__COMPOUND_URI__", compound_uri)
    try:
        results = g.query(query)
        rows = []
        for row in results:
            row_dict = {}
            for var in results.vars:
                val = row[var]
                if isinstance(val, URIRef):
                    row_dict[str(var)] = str(val)
                else:
                    row_dict[str(var)] = val
            rows.append(row_dict)
        return rows
    except Exception as e:
        print(f"Error for {compound_uri}: {e}")
        return []

# ---------------------------
# Collect all data (one row per compound)
# ---------------------------
def collect_all_data(g, limit=None):
    compounds = get_all_compounds(g)
    if limit:
        compounds = compounds[:limit]
    print(f"Processing {len(compounds)} compounds...")
    
    all_rows = []
    for uri in tqdm(compounds, desc="Compounds"):
        # ---- Query 1: Properties + bioactivities ----
        prop_rows = run_query_for_compound(g, uri, properties_template)
        if not prop_rows:
            row = {'compound': uri}
        else:
            base = prop_rows[0].copy()
            bio_labels = set()
            for r in prop_rows:
                if r.get('bioactivityLabel') and r['bioactivityLabel'] is not None:
                    bio_labels.add(r['bioactivityLabel'])
                for key in ['name', 'smiles', 'mw', 'formula', 'volume', 'monomass', 'nrotb', 'tpsa']:
                    if key in r and r[key] is not None and (key not in base or base[key] is None):
                        base[key] = r[key]
            base['bioactivities'] = " | ".join(sorted(bio_labels)) if bio_labels else None
            base.pop('bioactivityLabel', None)
            row = base
        
        # ---- Query 2: Location ----
        loc_rows = run_query_for_compound(g, uri, location_template)
        if loc_rows:
            species_set = set()
            state_set = set()
            city_set = set()
            for r in loc_rows:
                if r.get('speciesName'):
                    species_set.add(r['speciesName'])
                if r.get('stateName'):
                    state_set.add(r['stateName'])
                if r.get('cityName'):
                    city_set.add(r['cityName'])
            row['speciesName'] = " | ".join(sorted(species_set)) if species_set else None
            row['stateName']  = " | ".join(sorted(state_set))  if state_set else None
            row['cityName']   = " | ".join(sorted(city_set))   if city_set else None
        else:
            row['speciesName'] = None
            row['stateName']   = None
            row['cityName']    = None
        
        # ---- Query 3: Identifiers and extra descriptors ----
        id_rows = run_query_for_compound(g, uri, identifiers_template)
        if id_rows:
            # For scalar fields, take the first non-null value across rows
            for key in ['iupac', 'inchi', 'inchikey', 'hba', 'hbd', 'logp', 'lipinski']:
                for r in id_rows:
                    val = r.get(key)
                    if val is not None and val != '':
                        row[key] = val
                        break
                else:
                    row[key] = None
        else:
            for key in ['iupac', 'inchi', 'inchikey', 'hba', 'hbd', 'logp', 'lipinski']:
                row[key] = None
        
        all_rows.append(row)
    
    return pd.DataFrame(all_rows)

# ---------------------------
# Run and save
# ---------------------------
if __name__ == "__main__":
  #df_test = collect_all_data(g, limit=6)
  #print(df_test.T)
  # For full dataset, remove limit:
  df_full = collect_all_data(g)
  df_full.to_csv(os.path.join(DATA_RAW, "nubbe_full_dataset.csv"), index=False)
