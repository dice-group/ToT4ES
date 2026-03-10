import os
from rdflib import Graph, URIRef, Literal

def save_summary(dataset, entity_id, ranked, dataset_type="lmdb", k=5, out_dir="./outputs/out_summ_RELIN"):
    """
    saving relin to outputs folder.
    """
    os.makedirs(out_dir, exist_ok=True)
    base_data_dir = os.path.join(dataset, dataset_type+"_data")
    entity_dir = os.path.join(base_data_dir, str(entity_id))
    desc_path = os.path.join(entity_dir, f"{entity_id}_desc.nt")
    gold_path = os.path.join(entity_dir, f"{entity_id}_gold.nt")
    
    if not os.path.exists(desc_path):
        print(desc_path)
        print(f"MISSING DESC FILE FOR ENTITY {entity_id}")
        return
    
    # read desc to map predicate/object
    g = Graph()
    g.parse(desc_path, format="nt")

    # check subject
    subjects = [s for s, _, _ in g]
    main_subject = max(set(subjects), key=subjects.count)

    # save top-k triples
    out_entity_dir = os.path.join(out_dir, dataset_type, str(entity_id))
    os.makedirs(out_entity_dir, exist_ok=True)
    out_path = os.path.join(out_entity_dir, f"{entity_id}_top{k}.nt")

    with open(out_path, "w", encoding="utf-8") as f:
        for (p, o), score in ranked[:k]:
            # find matching triples
            matches = [t for t in g if t[1] == URIRef(p) and t[2] == o]
            if matches:
                s, p_, o_ = matches[0]
            else:
                # if no matching triples found, use main subject
                s = main_subject
                p_ = URIRef(p)
                o_ = o if isinstance(o, (URIRef, Literal)) else Literal(o)

            f.write(f"{s.n3()} {p_.n3()} {o_.n3()} .\n")

    # write rank_top5 and rank_top10 (same content)
    full_rank_path = os.path.join(out_entity_dir, f"{entity_id}_rank_top{k}.nt")
    
    with open(full_rank_path, "w", encoding="utf-8") as f:
        for (p, o), score in ranked:
            matches = [t for t in g if t[1] == URIRef(p) and t[2] == o]
            if matches:
                s, p_, o_ = matches[0]
            else:
                s = main_subject
                p_ = URIRef(p)
                o_ = o if isinstance(o, (URIRef, Literal)) else Literal(o)
            
            f.write(f"{s.n3()} {p_.n3()} {o_.n3()} # score={score:.4f}\n")
    
    # default: use full ranking if no gold found (fallback for ESBM/FACES)
    gold_count = len(ranked)

    if os.path.exists(gold_path):
        g_gold = Graph()
        g_gold.parse(gold_path, format="nt")
        gold_triples = [(p, o) for _, p, o in g_gold]
        gold_count = len(gold_triples)

    # write dynamic ranking (only ranked[:gold_count])
    dynamic_rank_path = os.path.join(out_entity_dir, f"{entity_id}_ranked_dynamic.nt")
    with open(dynamic_rank_path, "w", encoding="utf-8") as f:
        for (p, o), score in ranked[:gold_count]:
            # find matching triples
            matches = [t for t in g if t[1] == URIRef(p) and t[2] == o]
            if matches:
                s, p_, o_ = matches[0]
            else:
                # if no matching triples found, use main subject
                s = main_subject
                p_ = URIRef(p)
                o_ = o if isinstance(o, (URIRef, Literal)) else Literal(o)

            f.write(f"{s.n3()} {p_.n3()} {o_.n3()} .\n")
