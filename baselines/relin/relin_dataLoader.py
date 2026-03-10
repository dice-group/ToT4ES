import os
from rdflib import Graph
from collections import Counter

def load_entity_description(dataset_root:str, dataset_type:str, entity_id:str):
    """
    Load the RDF description of a specific entity from the dataset.
    """
    dataset_path = os.path.join(dataset_root, f"{dataset_type}_data/{entity_id}")
    desc_file = os.path.join(dataset_path, f"{entity_id}_desc.nt")

    # print(f"\n[INFO] Loading entity description:")
    # print(f"       Dataset path : {dataset_path}")
    # print(f"       Description file : {desc_file}")

    g = Graph()
    g.parse(desc_file, format="nt")
    
    subject_counts = Counter(s for s, _, _ in g)
    subject = max(subject_counts, key=lambda s: subject_counts[s])

    # print(f"[INFO] Detected main subject: {subject}")
    # print(f"[INFO] Subject triple count: {subject_counts[subject]} out of {len(g)} total")

    features = [(p, o) for s, p, o in g if s == subject]

    return g, subject, features