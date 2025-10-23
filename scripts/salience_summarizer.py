
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# See docstring in previous message; full script includes LLM scoring via transformers pipeline.

import argparse, json, re, math, yaml, sys, ast
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

def _lazy_import_transformers():
    try:
        import transformers
        return transformers
    except Exception:
        return None

def _build_llm_pipeline(model_id: str, device: str = "auto", dtype: Optional[str] = None, max_new_tokens: int = 256):
    transformers = _lazy_import_transformers()
    if transformers is None:
        return None
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=dtype if dtype else None)
    gen = pipeline("text-generation", model=model, tokenizer=tok, max_new_tokens=max_new_tokens,
                   do_sample=False, temperature=0.0, return_full_text=False)
    return gen

def load_yaml(path: Optional[str]) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        import yaml
        return yaml.safe_load(f) or {}

def parse_nt_line(line: str) -> Optional[Tuple[str, str, str]]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if not line.endswith("."):
        return None
    m = re.match(r'\s*<([^>]*)>\s+<([^>]*)>\s+(.*)\s*\.\s*$', line)
    if not m:
        return None
    s_iri, p_iri, o_raw = m.group(1), m.group(2), m.group(3).strip()
    o_iri_match = re.match(r'^<([^>]*)>$', o_raw)
    if o_iri_match:
        return s_iri, p_iri, f"<{o_iri_match.group(1)}>"
    lit_match = re.match(r'^"((?:[^"\\]|\\.)*)"(?:@([a-zA-Z\-]+))?(?:\^\^<[^>]*>)?$', o_raw)
    if lit_match:
        lit_text = lit_match.group(1)
        lit_text = lit_text.replace('\\"', '"').replace('\\\\', '\\').replace('\\n', '\n')
        return s_iri, p_iri, lit_text
    return s_iri, p_iri, o_raw

def load_triples_from_nt(path: Path) -> List[Dict[str, Any]]:
    triples = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        tid = 1
        for line in f:
            parsed = parse_nt_line(line)
            if not parsed:
                continue
            s, p, o = parsed
            triples.append({"id": f"t{tid}", "s": s, "p": p, "o": o})
            tid += 1
    return triples

def tail_from_iri(iri: str) -> str:
    iri = iri.strip("<>")
    if "#" in iri:
        return iri.rsplit("#", 1)[-1]
    return iri.rstrip("/").rsplit("/", 1)[-1]

class Canonicalizer:
    def __init__(self, cfg_pred: dict, cfg_ns: dict):
        self.alias_map = {}
        for canon, aliases in (cfg_pred.get("aliases") or {}).items():
            for a in aliases:
                self.alias_map[a.lower()] = canon.lower()
            self.alias_map[canon.lower()] = canon.lower()
        self.templates = (cfg_pred.get("templates") or {})
        self.ns = (cfg_ns.get("prefixes") or {})

    def shrink_curie(self, iri: str) -> str:
        iri = iri.strip("<>")
        for pref, base in self.ns.items():
            if iri.startswith(base):
                return f"{pref}:{iri[len(base):]}"
        return tail_from_iri(iri)

    def pred_tail_or_curie(self, p_iri: str) -> str:
        cur = self.shrink_curie(p_iri)
        return cur.split(":")[-1]

    def canonical_pred_key(self, p_iri: str) -> str:
        tail = self.pred_tail_or_curie(p_iri).lower()
        return self.alias_map.get(tail, tail)

def is_iri(value: str) -> bool:
    return isinstance(value, str) and value.startswith("<") and value.endswith(">")

def object_text(o: str) -> str:
    if is_iri(o):
        return tail_from_iri(o[1:-1]).replace("_", " ")
    return o

def detect_types(triples_for_entity: List[Dict[str, Any]], canon: Canonicalizer) -> List[str]:
    types = []
    for t in triples_for_entity:
        p_tail = canon.pred_tail_or_curie(t["p"]).lower()
        if p_tail in {"rdf:type", "type", "p31", "instanceof"}:
            if is_iri(t["o"]):
                types.append(canon.shrink_curie(t["o"]).split(":")[-1])
            else:
                types.append(str(t["o"]))
    seen, out = set(), []
    for x in types:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def learn_centrality(triples: List[Dict[str, Any]], canon: Canonicalizer,
                     top_k: int = 8, min_support: int = 25) -> Dict[str, set]:
    from collections import defaultdict, Counter
    subj_classes = defaultdict(set)
    for t in triples:
        p_tail = canon.pred_tail_or_curie(t["p"]).lower()
        if p_tail in {"rdf:type", "type", "p31", "instanceof"}:
            if is_iri(t["o"]):
                cls_tail = canon.shrink_curie(t["o"]).split(":")[-1]
            else:
                cls_tail = str(t["o"])
            subj_classes[t["s"]].add(cls_tail)

    bg_pred = Counter()
    class_pred = defaultdict(Counter)
    class_support = Counter()

    for t in triples:
        p_canon = canon.canonical_pred_key(t["p"])
        bg_pred[p_canon] += 1
        classes = subj_classes.get(t["s"]) or set()
        for c in classes:
            class_pred[c][p_canon] += 1
            class_support[c] += 1

    total_bg = sum(bg_pred.values()) or 1
    result = {}
    for c, pcnts in class_pred.items():
        if class_support[c] < min_support:
            continue
        sum_c = sum(pcnts.values()) or 1
        scored = []
        for p, cnt_c in pcnts.items():
            p_bg = bg_pred[p]
            p_c = cnt_c / sum_c
            p_bg_norm = p_bg / total_bg
            score = math.log((p_c / (p_bg_norm + 1e-12)) + 1e-12)
            scored.append((score, p))
        scored.sort(reverse=True)
        keep = [p for _, p in scored[:top_k]]
        result[c] = set(keep)
    return result

WEIGHTS = {"w_c": 0.35, "w_u": 0.25, "w_a": 0.25, "w_t": 0.15}

def authority_score(provenance: Optional[str]) -> float:
    if provenance:
        prov = provenance.lower()
        if any(dom in prov for dom in ["wikipedia", "dbpedia", "schema.org", "wikidata"]):
            return 0.9
        if any(dom in prov for dom in ["gov", "edu", "ac.", ".ac"]):
            return 0.8
        return 0.6
    return 0.5

def time_place_score(o: str, p_canon: str) -> float:
    otxt = object_text(o)
    has_year = bool(re.search(r'\b(18|19|20|21)\d{2}\b', otxt))
    has_date = bool(re.search(r'\b\d{1,2}\s*[A-Za-z]{3,9}\s*(18|19|20|21)\d{2}\b', otxt))
    has_coords = bool(re.search(r'[-+]?\d{1,3}\.\d+,\s*[-+]?\d{1,3}\.\d+', otxt))
    placey = bool(re.search(r'\b(city|town|province|state|county|country|region|district)\b', otxt, flags=re.I))
    if has_coords or has_date:
        return 1.0
    if has_year:
        return 0.9
    if p_canon in {"location", "city", "country"} or placey:
        return 0.6
    return 0.2

def uniqueness_score(o: str, p_tail: str, pred_freq: Dict[str, int]) -> float:
    score = 0.5
    otxt = object_text(o)
    if is_iri(o):
        score += 0.3
    if re.search(r'\b(19|20)\d{2}\b', otxt) or re.search(r'\d', otxt):
        score += 0.2
    if len(otxt) >= 15:
        score += 0.1
    freq = pred_freq.get(p_tail, 1)
    if freq <= 2:
        score += 0.1
    elif freq >= 6:
        score -= 0.1
    return max(0.0, min(1.0, score))

class CentralityResolver:
    def __init__(self, cfg_cent: dict, learned: Optional[Dict[str, set]] = None):
        self.cfg_cent = cfg_cent or {}
        self.learned = learned or {}
        self.cfg_classes = self.cfg_cent.get("classes") or {}

    def preds_for_class(self, cls_tail: str) -> set:
        if cls_tail in self.cfg_classes:
            return set([p.lower() for p in self.cfg_classes[cls_tail]])
        return self.learned.get(cls_tail, set())

    def score(self, p_canon: str, entity_types: List[str]) -> float:
        for et in entity_types:
            cent_set = self.preds_for_class(et)
            if p_canon in cent_set:
                return 1.0
            if et.lower().endswith("radiostation") and "radiostation" in (k.lower() for k in cent_set):
                return 1.0
        if p_canon in {"label","name","type","country","location","city","owner","homepage"}:
            return 0.8
        return 0.5

PROMPT_SYSTEM = """You are a careful salience scorer for entity summaries. Work only with the provided triples. Do not use external facts. Prefer triples that define the entity’s central identity, are distinctive, have strong provenance, or include time/place anchors. Think step-by-step privately, but output only the requested JSON."""

PROMPT_USER_TEMPLATE = """Inputs
entity_id: {entity_id}
entity_types: {entity_types}
triple: {triple}
entity_triples_context: {context}
allowed_relations: {allowed}

Task
Score this triple t for salience using:
salience(t) = 0.35*centrality + 0.25*uniqueness + 0.25*authority + 0.15*time_place_specificity

Guidelines
- centrality: is p prototypical for the given entity_types?
- uniqueness: does (p,o) make the entity stand out among entity_triples_context?
- authority: 0.9 for Wikipedia/DBpedia/Wikidata; 0.8 for .gov/.edu/.ac; else 0.6 if provenance present; 0.5 if missing.
- time_place_specificity: reward explicit dates/years/coordinates or precise locations; moderate credit for location predicates.
- Scores in [0,1]. No external facts. If allowed_relations present, set all scores to 0 if p tail not allowed.

Output JSON only
{
  "triple_id": "<id>",
  "salience_components": {
    "centrality": float,
    "uniqueness": float,
    "authority": float,
    "time_place_specificity": float
  },
  "salience": float
}"""

def _safe_parse_json(s: str) -> Optional[dict]:
    m = re.search(r'\{.*\}', s, flags=re.S)
    if not m:
        return None
    txt = m.group(0)
    try:
        return json.loads(txt)
    except Exception:
        try:
            import ast
            return ast.literal_eval(txt)
        except Exception:
            return None

def score_with_llm(gen, triple: dict, entity_id: str, entity_types: List[str],
                   context_triples: List[dict], allowed_relations: Optional[List[str]]):
    try:
        allowed = allowed_relations or []
        ctx = context_triples[:25]
        user_prompt = PROMPT_USER_TEMPLATE.format(
            entity_id=str(entity_id),
            entity_types=json.dumps(entity_types, ensure_ascii=False),
            triple=json.dumps(triple, ensure_ascii=False),
            context=json.dumps(ctx, ensure_ascii=False),
            allowed=json.dumps(allowed, ensure_ascii=False)
        )
        full_prompt = f"[SYSTEM]\n{PROMPT_SYSTEM}\n\n[USER]\n{user_prompt}\n\n[ASSISTANT]"
        out = gen(full_prompt, num_return_sequences=1)[0]["generated_text"]
        parsed = _safe_parse_json(out)
        if not parsed:
            raise ValueError("LLM did not return valid JSON")
        comp = parsed.get("salience_components", {})
        c = float(comp.get("centrality", 0.0))
        u = float(comp.get("uniqueness", 0.0))
        a = float(comp.get("authority", 0.0))
        tp = float(comp.get("time_place_specificity", 0.0))
        def clamp(x): return max(0.0, min(1.0, float(x)))
        c,u,a,tp = map(clamp, [c,u,a,tp])
        sal = 0.35*c + 0.25*u + 0.25*a + 0.15*tp
        return {"centrality": c, "uniqueness": u, "authority": a, "time_place_specificity": tp, "salience": sal}
    except Exception:
        return None

def jaccard_po_tokens(a_pred: str, a_obj: str, b_pred: str, b_obj: str) -> float:
    A = set(re.findall(r'\w+', (a_pred + " " + a_obj).lower()))
    B = set(re.findall(r'\w+', (b_pred + " " + b_obj).lower()))
    if not A and not B:
        return 1.0
    return len(A & B) / max(1, len(A | B))

def select_summary(candidates: List[Dict[str, Any]], k: int, canon: Canonicalizer) -> List[Dict[str, Any]]:
    selected = []
    used_pred_fams = set()
    for cand in candidates:
        p_alias = canon.canonical_pred_key(cand["triple"]["p"])
        if p_alias in used_pred_fams and len(selected) < k:
            continue
        selected.append(cand)
        used_pred_fams.add(p_alias)
        if len(selected) >= k:
            break
    if len(selected) < k:
        used_ids = {c["triple_id"] for c in selected}
        for cand in candidates:
            if cand["triple_id"] in used_ids:
                continue
            selected.append(cand)
            if len(selected) >= k:
                break
    return [
        {"order": i+1, "triple_id": c["triple_id"], "triple": c["triple"], "salience": c["salience"]}
        for i, c in enumerate(selected[:k])
    ]

def detect_primary_entity(triples: List[Dict[str, Any]]) -> str:
    subj_counts = Counter(t["s"] for t in triples)
    return subj_counts.most_common(1)[0][0]

def prepare_entity_label(entity_id: str, entity_triples: List[Dict[str, Any]], canon: Canonicalizer) -> str:
    label = None
    for t in entity_triples:
        if canon.pred_tail_or_curie(t["p"]).lower() in {"label","name","rdfs:label"}:
            label = object_text(t["o"])
            break
    if not label:
        if entity_id.startswith("http"):
            label = tail_from_iri(entity_id).replace("_", " ")
        else:
            label = entity_id
    return label

def build_candidates(entity_triples: List[Dict[str, Any]], canon: Canonicalizer,
                     cent, allowed_relations: Optional[List[str]],
                     use_llm: bool, llm_gen, entity_id: str):
    allowed = set([a.lower() for a in allowed_relations]) if allowed_relations else None
    entity_types = detect_types(entity_triples, canon)
    pred_freq = Counter(canon.pred_tail_or_curie(t["p"]).lower() for t in entity_triples)

    cands = []
    for t in entity_triples:
        p_tail = canon.pred_tail_or_curie(t["p"]).lower()
        p_canon = canon.canonical_pred_key(t["p"])
        if allowed and p_tail not in allowed and p_canon not in allowed:
            continue

        comp = None
        if use_llm and llm_gen is not None:
            comp = score_with_llm(
                llm_gen,
                triple={"id": t["id"], "s": t["s"], "p": t["p"], "o": t["o"], "provenance": t.get("provenance")},
                entity_id=entity_id,
                entity_types=entity_types,
                context_triples=[x for x in entity_triples if x["id"] != t["id"]],
                allowed_relations=list(allowed) if allowed else None,
            )
        if comp is None:
            c = cent.score(p_canon, entity_types)
            u = uniqueness_score(t["o"], p_tail, pred_freq)
            a = authority_score(t.get("provenance"))
            tp = time_place_score(t["o"], p_canon)
            sal = 0.35*c + 0.25*u + 0.25*a + 0.15*tp
            comp = {"centrality": c, "uniqueness": u, "authority": a, "time_place_specificity": tp, "salience": sal}

        cands.append({
            "triple_id": t["id"],
            "triple": {"s": t["s"], "p": t["p"], "o": t["o"]},
            "salience_components": {
                "centrality": round(comp["centrality"], 3),
                "uniqueness": round(comp["uniqueness"], 3),
                "authority": round(comp["authority"], 3),
                "time_place_specificity": round(comp["time_place_specificity"], 3),
            },
            "salience": round(comp["salience"], 4),
            "notes": "LLM-scored" if use_llm and llm_gen is not None else "heuristic-scored"
        })

    kept, seen = [], []
    for cand in sorted(cands, key=lambda x: x["salience"], reverse=True):
        p = canon.canonical_pred_key(cand["triple"]["p"])
        o = object_text(cand["triple"]["o"]).lower()
        dup = False
        for (pp, oo) in seen:
            j = jaccard_po_tokens(p, o, pp, oo)
            if j >= 0.8:
                dup = True
                break
        if not dup:
            kept.append(cand)
            seen.append((p, o))
    return kept, entity_types

def main():
    ap = argparse.ArgumentParser(description="Generalized salience triple summarizer (heuristics + optional LLM).")
    ap.add_argument("--nt", type=str, help="Path to N-Triples file")
    ap.add_argument("--json", type=str, help="Path to JSON input with entity triples")
    ap.add_argument("--entity-id", type=str, default=None, help="Force entity IRI/name (optional for NT mode)")
    ap.add_argument("--k", type=int, default=5, help="Number of triples to select (default 5)")
    ap.add_argument("--out", type=str, default=None, help="Save output JSON to this path")

    ap.add_argument("--centrality-yaml", type=str, default=None, help="centrality.yml path")
    ap.add_argument("--predicates-yaml", type=str, default=None, help="predicates.yml path")
    ap.add_argument("--namespaces-yaml", type=str, default=None, help="namespaces.yml path")

    ap.add_argument("--learn-centrality-from", type=str, default=None, help="Corpus NT to auto-learn class→central-preds")

    ap.add_argument("--use-llm", action="store_true", help="Use LLM for salience scoring")
    ap.add_argument("--llm-model-id", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="HF model ID")
    ap.add_argument("--llm-device", type=str, default="auto", help="device map (auto/cuda/cpu)")
    ap.add_argument("--llm-dtype", type=str, default=None, help="torch dtype (e.g., float16)")

    args = ap.parse_args()

    cfg_cent = load_yaml(args.centrality_yaml)
    cfg_pred = load_yaml(args.predicates_yaml)
    cfg_ns   = load_yaml(args.namespaces_yaml)
    canon = Canonicalizer(cfg_pred, cfg_ns)

    if args.nt:
        triples = load_triples_from_nt(Path(args.nt))
    elif args.json:
        with Path(args.json).open("r", encoding="utf-8") as f:
            data = json.load(f)
        triples = data.get("entity_description", [])
        if not triples:
            raise ValueError("JSON must contain non-empty 'entity_description'.")
    else:
        ap.error("Provide --nt or --json")

    learned = {}
    if args.learn_centrality_from:
        corpus_triples = load_triples_from_nt(Path(args.learn_centrality_from))
        top_k = (cfg_cent.get("defaults", {}) or {}).get("top_k", 8)
        min_support = (cfg_cent.get("defaults", {}) or {}).get("min_support", 25)
        learned = learn_centrality(corpus_triples, canon, top_k=top_k, min_support=min_support)

    cent = CentralityResolver(cfg_cent, learned)

    entity_id = args.entity_id or detect_primary_entity(triples)
    entity_triples = [t for t in triples if t["s"] == entity_id]
    entity_label = prepare_entity_label(entity_id, entity_triples, canon)

    allowed_relations = sorted({canon.pred_tail_or_curie(t["p"]).lower() for t in entity_triples})

    llm_gen = None
    if args.use_llm:
        llm_gen = _build_llm_pipeline(args.llm_model_id, device=args.llm_device, dtype=args.llm_dtype)

    candidates, type_hints = build_candidates(
        entity_triples, canon, cent,
        allowed_relations=allowed_relations,
        use_llm=args.use_llm, llm_gen=llm_gen, entity_id=entity_id
    )
    selected = select_summary(candidates, args.k, canon)

    out_obj = {
        "entity": entity_label,
        "k": args.k,
        "candidates": candidates,
        "selected_summary": selected,
        "meta": {
            "allowed_relations_applied": True,
            "type_hints_used": type_hints,
            "selection_rationale": "Greedy by salience with predicate-family diversity; dedup by Jaccard over predicate/object tokens.",
            "llm": {
                "enabled": bool(args.use_llm and llm_gen is not None),
                "model_id": args.llm_model_id if args.use_llm else None
            }
        }
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(json.dumps(out_obj, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()