import os
from rdflib import Graph, URIRef, Literal

def _build_raw_line_index(desc_path):
    """
    Build a lookup: (predicate_str, object_n3) -> original raw line from the .nt file.
    This preserves the exact original formatting (e.g. scientific notation in literals).
    """
    index = {}
    with open(desc_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Parse the raw line using rdflib to get structured (s, p, o)
            try:
                g_tmp = Graph()
                g_tmp.parse(data=line, format="nt")
                for s, p, o in g_tmp:
                    key = (str(p), o)
                    if key not in index:
                        index[key] = line
            except Exception:
                continue
    return index

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

    # build raw line index to preserve original formatting
    raw_index = _build_raw_line_index(desc_path)

    # check subject
    subjects = [s for s, _, _ in g]
    main_subject = max(set(subjects), key=subjects.count)

    def _get_raw_line(p, o, with_score=None):
        """Get the original raw .nt line for a (predicate, object) pair."""
        key = (str(p), o)
        raw = raw_index.get(key)
        if raw:
            # raw line already ends with ' .'
            if with_score is not None:
                # strip trailing ' .' and append score comment
                raw_stripped = raw.rstrip()
                if raw_stripped.endswith(" ."):
                    raw_stripped = raw_stripped[:-2].rstrip()
                return f"{raw_stripped} # score={with_score:.4f}\n"
            return raw + "\n" if not raw.endswith("\n") else raw
        else:
            # fallback to rdflib serialization
            matches = [t for t in g if t[1] == URIRef(p) and t[2] == o]
            if matches:
                s, p_, o_ = matches[0]
            else:
                s = main_subject
                p_ = URIRef(p)
                o_ = o if isinstance(o, (URIRef, Literal)) else Literal(o)
            line = f"{s.n3()} {p_.n3()} {o_.n3()} ."
            if with_score is not None:
                return f"{s.n3()} {p_.n3()} {o_.n3()} # score={with_score:.4f}\n"
            return line + "\n"

    # save top-k triples
    out_entity_dir = os.path.join(out_dir, dataset_type, str(entity_id))
    os.makedirs(out_entity_dir, exist_ok=True)
    out_path = os.path.join(out_entity_dir, f"{entity_id}_top{k}.nt")

    with open(out_path, "w", encoding="utf-8") as f:
        for (p, o), score in ranked[:k]:
            f.write(_get_raw_line(p, o))

    # write rank_top5 and rank_top10 (same content)
    full_rank_path = os.path.join(out_entity_dir, f"{entity_id}_rank_top{k}.nt")
    
    with open(full_rank_path, "w", encoding="utf-8") as f:
        for (p, o), score in ranked:
            f.write(_get_raw_line(p, o, with_score=score))
    
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
            f.write(_get_raw_line(p, o))
