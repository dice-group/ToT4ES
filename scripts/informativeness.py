#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entity Informativeness Summarizer (LLM-owned scores, Robust JSON Repair)
with Diversity Policies + Rich Outputs + NT export + Batch Logs
----------------------------------------------------------------
- Input: N-Triples (.nt) file OR a directory of .nt files (recursive), or JSON {"entity_description":[...]}.
- Learns generic types (IDF-style) + predicate families (seeds + subject-set Jaccard + name similarity).
- Summarizes context: type_rollup (drop generic types), predicate_histogram, predicate_examples, predicate_families.

Scoring (no "relatedness" anymore):
- informativeness: LLM returns informativeness and components:
    * stat_info  : statistical rarity/utility (IDF-ish, count-based rarity, novelty of object tokens)
    * onto_info  : ontological specificity (type specificity, distance from generic, role-centricity)
  In hybrid mode we expect the LLM to combine them. We can also locally blend as fallback.

Selection diversity
- --diversity none       : pure top-k by informativeness (no uniqueness constraints)
- --diversity predicate  : at most one per predicate (default)
- --diversity family     : at most one per learned family (aliases merged).
  Tip: allow more types with --family-limit "type=2,*=1" (only used with --diversity family)

Outputs
- --out (JSON), --emit-csv, --emit-md, --emit-nt, --pretty-console
  Note: literals lose @lang/^^datatype metadata because the parser normalizes to plain strings.

New / changed flags
- --info-mode {statistical,ontological,hybrid}: which criterion the LLM optimizes (default: hybrid)
- --beta: local fallback blend i = beta*onto + (1-beta)*stat (only if LLM omits informativeness)
- --fallback-local-stat: compute stat_info from predicate_histogram if LLM omits it
- --fallback-local-onto: compute onto_info from type_rollup & generic_types if LLM omits it
- --log-dir: save batch logs (prompts, raw, parsed) under this directory
"""

import argparse, json, re, sys, time, csv, hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime

# =============================== Logging ===================================

class LogSink:
    def __init__(self, root: Optional[str], run_meta: Dict[str, Any]):
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        rid = run_meta.get("run_id", "run")
        base = Path(root) if root else Path("logs") / f"run-{ts}-{rid}"
        self.base = base
        self.base.mkdir(parents=True, exist_ok=True)
        with (self.base / "run_meta.json").open("w", encoding="utf-8") as f:
            json.dump(run_meta, f, ensure_ascii=False, indent=2)

    def _batch_dir(self, batch_idx: int, attempt: int) -> Path:
        d = self.base / f"batch-{batch_idx:04d}-attempt-{attempt}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def log_batch(self,
                  batch_idx: int,
                  attempt: int,
                  system_prompt: str,
                  user_prompt: str,
                  raw_text: Optional[str],
                  parsed_obj: Optional[List[dict]],
                  extra_meta: Optional[Dict[str, Any]] = None):
        d = self._batch_dir(batch_idx, attempt)
        (d / "system.txt").write_text(system_prompt or "", encoding="utf-8")
        (d / "user.txt").write_text(user_prompt or "", encoding="utf-8")
        if raw_text is not None:
            (d / "raw.txt").write_text(raw_text, encoding="utf-8")
        if parsed_obj is not None:
            with (d / "parsed.json").open("w", encoding="utf-8") as f:
                json.dump(parsed_obj, f, ensure_ascii=False, indent=2)
        meta = {"batch_index": batch_idx, "attempt": attempt}
        if extra_meta:
            meta.update(extra_meta)
        with (d / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

# =========================== Transformers (LLM) ===========================

def _lazy_import_transformers():
    try:
        import transformers  # noqa
        return transformers
    except Exception:
        return None

def build_llm_pipeline(model_id: str, device: str = "auto", dtype: Optional[str] = None,
                       max_new_tokens: int = 1024):
    transformers = _lazy_import_transformers()
    if transformers is None:
        raise RuntimeError("Transformers not installed. `pip install transformers accelerate torch`")
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=dtype if dtype else None
    )
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        return_full_text=False
    )
    return gen, tok

def apply_chat_template(tokenizer, system: str, user: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role":"system","content":system},
             {"role":"user",  "content":user}],
            tokenize=False,
            add_generation_prompt=True
        )
    return f"[SYSTEM]\n{system}\n\n[USER]\n{user}\n\n[ASSISTANT]"

def extract_text(gen_out):
    if not gen_out or not isinstance(gen_out, list):
        raise ValueError(f"Unexpected pipeline output: {type(gen_out)} -> {gen_out}")
    first = gen_out[0]
    if isinstance(first, dict):
        return first.get("generated_text") or first.get("text") or json.dumps(first, ensure_ascii=False)
    return str(first)

def clamp01(x: float) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

# =========================== Parsing & utilities ===========================

NT_TRIPLE_RE = re.compile(r'\s*<([^>]*)>\s+<([^>]*)>\s+(.*)\s*\.\s*$')
IRI_ONLY_RE = re.compile(r'^<([^>]*)>$')
LITERAL_RE  = re.compile(r'^"((?:[^"\\]|\\.)*)"(?:@([a-zA-Z\-]+))?(?:\^\^<[^>]*>)?$')

def parse_nt_line(line: str) -> Optional[Tuple[str,str,str]]:
    line = line.strip()
    if not line or line.startswith("#"): return None
    if not line.endswith("."): return None
    m = NT_TRIPLE_RE.match(line)
    if not m: return None
    s, p, o_raw = m.group(1), m.group(2), m.group(3).strip()
    mi = IRI_ONLY_RE.match(o_raw)
    if mi:
        return s, p, f"<{mi.group(1)}>"
    ml = LITERAL_RE.match(o_raw)
    if ml:
        lit = ml.group(1)
        lit = lit.replace('\\"','"').replace('\\\\','\\').replace('\\n','\n')
        return s, p, lit  # note: lang/datatype dropped
    return s, p, o_raw

def load_triples_from_nt(path: Path) -> List[Dict[str, Any]]:
    triples = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        tid = 1
        for line in f:
            t = parse_nt_line(line)
            if not t: continue
            s, p, o = t
            triples.append({"id": f"t{tid}", "s": s, "p": p, "o": o})
            tid += 1
    return triples

def load_triples_from_nt_dir(path: Path) -> List[Dict[str, Any]]:
    triples = []
    tid = 1
    for f in sorted(path.rglob("*.nt")):
        with f.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                t = parse_nt_line(line)
                if not t:
                    continue
                s, p, o = t
                triples.append({"id": f"t{tid}", "s": s, "p": p, "o": o})
                tid += 1
    return triples

def is_iri(x: str) -> bool:
    return isinstance(x, str) and x.startswith("<") and x.endswith(">")

def iri_tail(iri: str) -> str:
    iri = iri.strip("<>")
    if "#" in iri:
        return iri.rsplit("#",1)[-1]
    return iri.rstrip("/").rsplit("/",1)[-1]

def obj_preview(o: str, limit: int = 60) -> str:
    if is_iri(o):
        return iri_tail(o).replace("_"," ")
    s = o.replace("\n"," ").strip()
    return s if len(s) <= limit else s[:limit-1] + "…"

def pred_tail(p: str) -> str:
    p = p.strip("<>")
    if "#" in p: return p.rsplit("#",1)[-1].lower()
    return p.rstrip("/").rsplit("/",1)[-1].lower()

def detect_primary_entity(triples: List[Dict[str,Any]]) -> str:
    return Counter(t["s"] for t in triples).most_common(1)[0][0]

def detect_types(entity_triples: List[Dict[str,Any]]) -> List[str]:
    out, seen = [], set()
    for t in entity_triples:
        if pred_tail(t["p"]) in {"rdf:type","type","p31","instanceof"}:
            tail = iri_tail(t["o"]) if is_iri(t["o"]) else str(t["o"])
            if tail not in seen:
                out.append(tail); seen.add(tail)
    return out

def label_for_entity(entity_id: str, entity_triples: List[Dict[str,Any]]) -> str:
    for t in entity_triples:
        if pred_tail(t["p"]) in {"rdfs:label","label","foaf:name","name"}:
            return obj_preview(t["o"], limit=120)
    if entity_id.startswith("http"):
        return iri_tail(entity_id).replace("_"," ")
    return entity_id

# ===================== Learning generic types & families ====================

def norm_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def learn_generic_types(
    triples: List[Dict[str, Any]],
    type_predicates: Tuple[str, ...] = ("rdf:type","type","p31","instanceof"),
    min_entities: int = 500,
    df_frac_threshold: float = 0.20,
    min_df: int = 50
) -> Set[str]:
    entities = set(t["s"] for t in triples)
    if not entities:
        return set()

    type_to_subjects = defaultdict(set)
    for t in triples:
        if pred_tail(t["p"]) in type_predicates:
            typ = iri_tail(t["o"]) if is_iri(t["o"]) else str(t["o"])
            type_to_subjects[typ].add(t["s"])

    total = len(entities)
    generic = set()
    for typ, subs in type_to_subjects.items():
        df = len(subs)
        if total >= min_entities:
            if df >= min_df and (df / total) >= df_frac_threshold:
                generic.add(typ)
        else:
            if df >= max(min_df, 10) and (df / total) >= max(df_frac_threshold, 0.35):
                generic.add(typ)
    always = {"Thing","owl:Thing","dul:Agent","Agent","Entity","Resource"}
    return generic | (always & set(type_to_subjects.keys()))

def _seed_family_name(pt: str) -> str:
    n = norm_token(pt)
    if re.search(r"(label|name|preflabel|title)", n): return "label"
    if re.search(r"(type|instance|class)", n): return "type"
    if re.search(r"(home ?page|website|officialwebsite|url)", n): return "homepage"
    if re.search(r"(sameas|equivalent|exact ?match)", n): return "sameAs"
    if re.search(r"(location|place|city|country|state|region|headquarter|hqlocation)", n): return "location"
    if re.search(r"(birth|death|founded|inception|start|end|opened|closed|established)", n): return "timeEvent"
    return ""

def learn_predicate_families(
    triples: List[Dict[str, Any]],
    max_families: int = 24,
    min_support: int = 30,
    jaccard_threshold: float = 0.75,
    name_similarity_threshold: float = 0.75
) -> Dict[str, List[str]]:
    pred_subjects = defaultdict(set)
    pred_counts = Counter()
    for t in triples:
        pt = pred_tail(t["p"])
        pred_subjects[pt].add(t["s"])
        pred_counts[pt] += 1

    frequent_preds = [p for p,c in pred_counts.items() if c >= min_support]

    # union-find
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    def union(a,b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # seeds
    for p in frequent_preds:
        fam = _seed_family_name(p)
        if fam:
            union(fam, p)

    # subject-set Jaccard
    fps = list(frequent_preds)
    for i in range(len(fps)):
        p1, s1 = fps[i], pred_subjects[fps[i]]
        if not s1: continue
        for j in range(i+1, len(fps)):
            p2, s2 = fps[j], pred_subjects[fps[j]]
            if not s2: continue
            inter = len(s1 & s2)
            if inter == 0: continue
            jac = inter / float(len(s1 | s2))
            if jac >= jaccard_threshold:
                union(p1, p2)

    # local-name similarity (cheap)
    def name_sim(a: str, b: str) -> float:
        A = set(re.findall(r"[a-z]+", a))
        B = set(re.findall(r"[a-z]+", b))
        if not A and not B: return 1.0
        return len(A & B) / max(1, len(A | B))

    for i in range(len(fps)):
        a = fps[i]
        for j in range(i+1, len(fps)):
            b = fps[j]
            if name_sim(a, b) >= name_similarity_threshold:
                union(a, b)

    groups = defaultdict(list)
    for p in frequent_preds:
        groups[find(p)].append(p)

    seeded_names = {"label","type","homepage","sameAs","location","timeEvent"}
    families = {}
    for root, members in groups.items():
        fam_name = root if root in seeded_names else max(members, key=lambda x: pred_counts[x])
        families[fam_name] = sorted(members)

    fam_scored = sorted(
        families.items(),
        key=lambda kv: sum(pred_counts[p] for p in kv[1]),
        reverse=True
    )
    families = dict(fam_scored[:max_families])

    minimal = {
        "label": ["rdfs:label","foaf:name","skos:prefLabel","name"],
        "type":  ["rdf:type","type","p31","instanceof"],
        "homepage": ["foaf:homepage","schema:url","url","homepage","officialWebsite","website"]
    }
    for fam, plist in minimal.items():
        families.setdefault(fam, [])
        for p in plist:
            if p not in families[fam]:
                families[fam].append(p)
        families[fam] = sorted(set(families[fam]))

    return families

# ======================= Context summarization (learned) ====================

def filter_specific_types(types: List[str], generic_types: Set[str], max_keep: int = 5) -> List[str]:
    scored = []
    for t in types:
        base = t.split(":")[-1]
        is_generic = (t in generic_types) or (base in generic_types)
        score = (0 if is_generic else 1) + 0.001 * len(base)
        scored.append((score, t))
    scored.sort(reverse=True)
    return [t for s,t in scored[:max_keep] if s > 0]

def build_predicate_histogram(entity_triples: List[Dict[str,Any]], top_n: int = 20) -> Dict[str,int]:
    cnt = Counter(pred_tail(t["p"]) for t in entity_triples)
    return dict(cnt.most_common(top_n))

def build_predicate_examples(entity_triples: List[Dict[str,Any]], per_pred: int = 2) -> Dict[str, List[str]]:
    buckets: Dict[str, List[str]] = defaultdict(list)
    for t in entity_triples:
        pt = pred_tail(t["p"])
        if len(buckets[pt]) < per_pred:
            buckets[pt].append(obj_preview(t["o"]))
    return dict(buckets)

def allowed_relations_default(entity_triples: List[Dict[str,Any]]) -> List[str]:
    return sorted({pred_tail(t["p"]) for t in entity_triples})

# =============================== Prompts (batch) ============================

SYSTEM_INFO = (
    "You are an entity summarization scorer. Work strictly with the provided inputs. "
    "Return valid JSON ONLY — no prose, no backticks, no explanations.\n\n"
    "Goal: score each triple by its informativeness in [0,1]. Also return components:\n"
    "- stat_info: statistical utility/rarity (IDF-like) based on predicate_histogram and object distinctiveness.\n"
    "- onto_info: ontological specificity/identity (specific types vs generic; role centrality for the entity).\n"
    "Interpretation hints:\n"
    "- Statistical: Prefer predicates that are less frequent for THIS entity, or whose objects add new facts "
    "(e.g., the first homepage, one canonical label, unique identifiers). Normalize to [0,1].\n"
    "- Ontological: Prefer type- or role-defining facts (specific rdf:type, membership, core roles), "
    "downweight generic types (from learned_generic_types). Normalize to [0,1].\n"
    "- Hybrid: Combine both (you choose a reasonable combination consistent with the mode).\n"
    "Clamp all values to [0,1]. No external knowledge.\n"
    "Output format strictly:\n"
    '[{"triple_id":"<id>","info_components":{"stat_info":float,"onto_info":float},"informativeness":float}]'
)

USER_INFO_TEMPLATE = (
    "Scoring mode: {INFO_MODE}\n\n"
    "Entity:\n"
    "- id: {ENTITY_ID}\n"
    "- preferred_label: {ENTITY_LABEL}\n"
    "- type_rollup: {TYPE_ROLLUP}\n\n"
    "Context summary:\n"
    "- predicate_histogram: {PRED_HISTOGRAM}\n"
    "- predicate_families: {PREDICATE_FAMILIES}\n"
    "- predicate_examples: {PREDICATE_EXAMPLES}\n"
    "- learned_generic_types: {GENERIC_TYPES}\n\n"
    "Triples_to_score:\n"
    "{TRIPLES_LIST}\n\n"
    "Constraints:\n"
    "- allowed_relations: {ALLOWED_LIST}\n\n"
    "Return JSON array with keys 'triple_id', 'info_components', 'informativeness'."
)

def parse_or_repair_json_array(txt: str):
    if txt is None:
        return None
    start = txt.find('[')
    if start == -1:
        return None
    chunk = txt[start:]
    try:
        obj = json.loads(chunk)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    objs = re.findall(r'\{[^{}]*\}', chunk, flags=re.S)
    items = []
    for o in objs:
        try:
            items.append(json.loads(o))
        except Exception:
            continue
    return items if items else None

# =============================== Scoring (batch) ============================

def batch_score(gen, tok, system_prompt: str, user_text: str,
                debug: bool,
                retries: int,
                delay: float,
                logger: Optional[LogSink],
                batch_idx: int,
                extra_meta: Dict[str, Any]) -> Optional[List[dict]]:
    raw_text = None
    parsed = None
    for attempt in range(retries+1):
        try:
            prompt = apply_chat_template(tok, system_prompt, user_text)
            out = gen(prompt, num_return_sequences=1)
            raw_text = extract_text(out)
            parsed = parse_or_repair_json_array(raw_text)
            if logger:
                logger.log_batch(
                    batch_idx=batch_idx,
                    attempt=attempt,
                    system_prompt=system_prompt,
                    user_prompt=user_text,
                    raw_text=raw_text,
                    parsed_obj=parsed,
                    extra_meta=extra_meta
                )
            if parsed is None or not isinstance(parsed, list):
                raise ValueError("LLM did not return a JSON array")
            norm = [x for x in parsed if isinstance(x, dict)]
            return norm
        except Exception as e:
            if debug:
                print(f"[batch] attempt {attempt} error: {e}", file=sys.stderr)
                print("RAW_LLM_OUT_START", raw_text if raw_text is not None else "", "RAW_LLM_OUT_END", file=sys.stderr)
            time.sleep(delay)
    return None

# ========================= Local fallback (optional) ========================

def local_stat_for_pred(ptail: str, pred_hist: Dict[str,int]) -> float:
    # Higher when rarer within this entity (inverse of frequency)
    if not pred_hist:
        return 0.0
    c = pred_hist.get(ptail, 0)
    m = max(pred_hist.values()) if pred_hist else 0
    if m <= 0: return 0.0
    # normalize rarity: 1 - freq
    return clamp01(1.0 - (c / m))

def local_onto_for_pred(ptail: str, types_specific: List[str], generic_types: Set[str]) -> float:
    # Cheap proxy:
    # - reward type/role-like predicates
    # - otherwise neutral 0.5
    n = norm_token(ptail)
    base = 0.5
    if re.search(r"(type|instance|class|occupation|position|role)$", n):
        # more specific types => more informative
        spec_boost = min(1.0, 0.4 + 0.1*len([t for t in types_specific if t not in generic_types]))
        return clamp01(0.6 + spec_boost)
    if re.search(r"(label|name|identifier|id)$", n):
        return 0.7
    if re.search(r"(homepage|url|website)$", n):
        return 0.6
    if re.search(r"(birth|death|founded|inception|start|end)$", n):
        return 0.6
    return base

# =============================== Selection & dedup ==========================

def jaccard_po_tokens(a_pred: str, a_obj: str, b_pred: str, b_obj: str) -> float:
    A = set(re.findall(r'\w+', (a_pred + " " + a_obj).lower()))
    B = set(re.findall(r'\w+', (b_pred + " " + b_obj).lower()))
    if not A and not B: return 1.0
    return len(A & B) / max(1, len(A | B))

def select_summary_predicate(candidates: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    selected, seen_preds = [], set()
    for c in sorted(candidates, key=lambda x: x["informativeness"], reverse=True):
        p = pred_tail(c["triple"]["p"])
        if p in seen_preds and len(selected) < k:
            continue
        selected.append(c)
        seen_preds.add(p)
        if len(selected) >= k:
            break
    if len(selected) < k:
        used = {x["triple_id"] for x in selected}
        for c in candidates:
            if c["triple_id"] in used: continue
            selected.append(c)
            if len(selected) >= k: break
    return [{"order": i+1, "triple_id": c["triple_id"], "triple": c["triple"], "informativeness": c["informativeness"]}
            for i, c in enumerate(selected[:k])]

def select_summary_family(candidates: List[Dict[str, Any]], k: int,
                          family_of_pred: Dict[str,str],
                          per_family_limit: Dict[str,int]) -> List[Dict[str, Any]]:
    def limit_for(fam: str) -> int:
        return per_family_limit.get(fam, per_family_limit.get("*", 1))
    selected, fam_counts, used = [], defaultdict(int), set()
    for c in sorted(candidates, key=lambda x: x["informativeness"], reverse=True):
        p = pred_tail(c["triple"]["p"]).lower()
        fam = family_of_pred.get(p, f"pred:{p}")
        if fam_counts[fam] >= limit_for(fam) and len(selected) < k:
            continue
        selected.append(c); fam_counts[fam] += 1; used.add(c["triple_id"])
        if len(selected) >= k: break
    if len(selected) < k:
        for c in candidates:
            if c["triple_id"] in used: continue
            selected.append(c); used.add(c["triple_id"])
            if len(selected) >= k: break
    return [{"order": i+1, "triple_id": c["triple_id"], "triple": c["triple"], "informativeness": c["informativeness"]}
            for i, c in enumerate(selected[:k])]

def select_topk_no_diversity(candidates: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    topk = sorted(candidates, key=lambda x: x["informativeness"], reverse=True)[:k]
    return [{"order": i+1, "triple_id": c["triple_id"], "triple": c["triple"], "informativeness": c["informativeness"]}
            for i, c in enumerate(topk)]

# =============================== Output helpers ============================

def ensure_out_dir(out_path: Optional[str], out_dir: Optional[str]) -> Path:
    if out_dir:
        p = Path(out_dir)
    elif out_path:
        p = Path(out_path).parent
    else:
        p = Path(".")
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_json(out_obj: dict, out_path: Optional[str]):
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(out_path).open("w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(out_obj, ensure_ascii=False, indent=2))

def write_candidates_csv(cands: List[Dict[str, Any]], dest: Path):
    fp = dest / "candidates.csv"
    with fp.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["triple_id","s","p","o","stat_info","onto_info","informativeness","llm_informativeness"])
        for c in cands:
            s = c["triple"]["s"]; p = c["triple"]["p"]; o = c["triple"]["o"]
            stat = c["info_components"]["stat_info"]
            onto = c["info_components"]["onto_info"]
            inf = c["informativeness"]
            llm_i = c["info_components"].get("llm_informativeness", "")
            w.writerow([c["triple_id"], s, p, o, stat, onto, inf, llm_i if llm_i != None else ""])
    return fp

def write_selected_csv(sel: List[Dict[str, Any]], dest: Path):
    fp = dest / "selected.csv"
    with fp.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order","triple_id","s","p","o","informativeness"])
        for x in sel:
            w.writerow([x["order"], x["triple"]["s"], x["triple"]["p"], x["triple"]["o"], x["informativeness"]])
    return fp

def nt_escape_literal(s: str) -> str:
    return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

def write_selected_nt(sel: List[Dict[str, Any]], dest: Path, filename: Optional[str] = None):
    fp = dest / (filename or "selected.nt")
    with fp.open("w", encoding="utf-8") as f:
        for x in sel:
            s = x["triple"]["s"]; p = x["triple"]["p"]; o = x["triple"]["o"]
            s_out = f"<{s}>" if not s.startswith("<") else s
            p_out = f"<{p}>" if not p.startswith("<") else p
            o_out = o if is_iri(o) else f"\"{nt_escape_literal(str(o))}\""
            f.write(f"{s_out} {p_out} {o_out} .\n")
    return fp

def write_markdown_report(entity_label: str, info_mode: str, meta: dict, candidates: List[Dict[str,Any]], selected: List[Dict[str,Any]], dest: Path):
    fp = dest / "report.md"
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    def md_escape(s: str) -> str:
        return str(s).replace("|","\\|")
    with fp.open("w", encoding="utf-8") as f:
        f.write(f"# Entity summary report\n\n")
        f.write(f"- entity: {entity_label}\n- info_mode: {info_mode}\n- generated: {ts}\n\n")
        f.write("## type rollup\n\n")
        f.write((", ".join(meta.get("type_rollup", [])) or "(none)") + "\n\n")
        f.write("## predicate histogram (top)\n\n")
        if meta.get("predicate_histogram"):
            f.write("| predicate | count |\n|---|---|\n")
            for p, c in meta["predicate_histogram"].items():
                f.write(f"| {md_escape(p)} | {c} |\n")
            f.write("\n")
        else:
            f.write("(none)\n\n")
        f.write("## selected summary\n\n")
        f.write("| # | predicate | object (preview) | informativeness |\n|---|---|---|---|\n")
        for x in selected:
            p_tail = pred_tail(x["triple"]["p"])
            o_prev = obj_preview(x["triple"]["o"])
            f.write(f"| {x['order']} | {md_escape(p_tail)} | {md_escape(o_prev)} | {x['informativeness']:.4f} |\n")
        f.write("\n## top candidates (up to 30)\n\n")
        f.write("| rank | triple_id | predicate | object (preview) | stat | onto | informativeness | llm_informativeness |\n|---|---|---|---|---:|---:|---:|---:|\n")
        for i, c in enumerate(sorted(candidates, key=lambda y: y["informativeness"], reverse=True)[:30], 1):
            p_tail = pred_tail(c["triple"]["p"])
            o_prev = obj_preview(c["triple"]["o"])
            stat = c["info_components"]["stat_info"]
            onto = c["info_components"]["onto_info"]
            inf  = c["informativeness"]
            llm_i = c["info_components"].get("llm_informativeness", None)
            f.write(f"| {i} | {md_escape(c['triple_id'])} | {md_escape(p_tail)} | {md_escape(o_prev)} | {stat:.3f} | {onto:.3f} | {inf:.4f} | {(f'{llm_i:.4f}' if isinstance(llm_i,(int,float)) else '')} |\n")
        f.write("\n## predicate families (learned)\n\n")
        fams = meta.get("predicate_families", {})
        if fams:
            for fam, plist in fams.items():
                f.write(f"- {fam}: {', '.join(plist)}\n")
            f.write("\n")
        else:
            f.write("(none)\n\n")
        f.write("## config\n\n")
        f.write(f"- model_id: {meta.get('model_id','')}\n- batch_size: {meta.get('batch_size')}\n- max_new_tokens: {meta.get('max_new_tokens')}\n")
        gtypes = meta.get("learned_generic_types", [])
        if gtypes:
            f.write(f"- learned_generic_types: {', '.join(gtypes[:50])}")
            if len(gtypes) > 50:
                f.write(f" (+{len(gtypes)-50} more)")
            f.write("\n")
    return fp

def print_pretty_console(entity: str, selected: List[Dict[str,Any]], candidates: List[Dict[str,Any]], pred_hist: Dict[str,int]):
    def line():
        print("-"*98)
    print(f"\nEntity: {entity}")
    line()
    print("Selected summary (k={}):".format(len(selected)))
    print("{:>2}  {:<18}  {:<55}  {:>9}".format("#","predicate","object (preview)","info"))
    line()
    for x in selected:
        p = pred_tail(x["triple"]["p"])[:18]
        o = obj_preview(x["triple"]["o"], 55)
        print("{:>2}  {:<18}  {:<55}  {:>9.4f}".format(x["order"], p, o, x["informativeness"]))
    line()
    print("Top candidates:")
    print("{:>3} {:<8} {:<18} {:<46} {:>8} {:>8} {:>9}".format("rk","id","predicate","object (preview)","stat","onto","info"))
    line()
    for i, c in enumerate(sorted(candidates, key=lambda y: y["informativeness"], reverse=True)[:15], 1):
        p = pred_tail(c["triple"]["p"])[:18]
        o = obj_preview(c["triple"]["o"], 46)
        st = c["info_components"]["stat_info"]
        on = c["info_components"]["onto_info"]
        inf = c["informativeness"]
        print("{:>3} {:<8} {:<18} {:<46} {:>8.3f} {:>8.3f} {:>9.4f}".format(i, c["triple_id"], p, o, st, on, inf))
    line()
    if pred_hist:
        print("Predicate histogram (top):")
        print("{:<22} {:>6}".format("predicate","count"))
        for p, cnt in list(pred_hist.items())[:15]:
            print("{:<22} {:>6}".format(p[:22], cnt))
        line()

def infer_dataset_id(args) -> str:
    if args.nt:
        p = Path(args.nt)
        if p.exists() and p.is_dir():
            return p.name
        return p.parent.name or p.stem.split("_")[0]
    if args.json:
        p = Path(args.json)
        return p.parent.name or p.stem.split("_")[0]
    return "summary"

# =============================== Main ======================================

def parse_family_limit(s: Optional[str]) -> Dict[str,int]:
    """
    Parse --family-limit like "type=2,*=1". Default {'*':1}
    Only used when --diversity family.
    """
    limits = {'*': 1}
    if not s:
        return limits
    for part in s.split(","):
        part = part.strip()
        if not part or "=" not in part: continue
        k, v = part.split("=", 1)
        k = k.strip(); v = v.strip()
        try:
            limits[k] = int(v)
        except Exception:
            pass
    if "*" not in limits:
        limits["*"] = 1
    return limits

def main():
    ap = argparse.ArgumentParser(description="Entity Informativeness Summarizer with diversity policies, rich outputs, and batch logs (LLM-owned scores).")
    ap.add_argument("--nt", type=str, help="Path to an N-Triples file OR a directory containing .nt files (recursively)")
    ap.add_argument("--json", type=str, help="Path to JSON with key 'entity_description'")
    ap.add_argument("--entity-id", type=str, default=None, help="Override entity subject IRI")
    ap.add_argument("--k", type=int, default=5, help="Number of triples to select")

    ap.add_argument("--info-mode", choices=["statistical","ontological","hybrid"], default="hybrid",
                    help="Statistical: rarity/utility only; Ontological: specificity/identity; Hybrid: combine both.")
    ap.add_argument("--beta", type=float, default=0.6,
                    help="Local fallback weight for onto vs stat when LLM omits informativeness: i = beta*onto + (1-beta)*stat.")

    ap.add_argument("--diversity", choices=["none","predicate","family"], default="predicate",
                    help="Selection policy: none=pure top-k; predicate=one per predicate; family=one per learned family.")
    ap.add_argument("--family-limit", type=str, default="*==1",
                    help='Only used when --diversity family. Comma-separated limits, e.g., "type=2,*=1"')

    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--topn-predicates", type=int, default=20)
    ap.add_argument("--llm-model-id", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--llm-device", type=str, default="auto")
    ap.add_argument("--llm-dtype", type=str, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=1024)

    ap.add_argument("--out", type=str, default=None, help="Write full JSON to this file (prints to stdout if omitted)")
    ap.add_argument("--out-dir", type=str, default=None, help="Directory for CSV/MD/NT; defaults to dirname(--out) or '.'")
    ap.add_argument("--emit-csv", action="store_true", help="Write candidates.csv and selected.csv")
    ap.add_argument("--emit-md", action="store_true", help="Write report.md")
    ap.add_argument("--emit-nt", action="store_true", help="Write selected.nt (and rank file)")
    ap.add_argument("--pretty-console", action="store_true", help="Print compact tables")
    ap.add_argument("--debug", action="store_true")

    # learners
    ap.add_argument("--min-entities-generic", type=int, default=500)
    ap.add_argument("--df-frac-threshold", type=float, default=0.20)
    ap.add_argument("--min-df-generic", type=int, default=50)
    ap.add_argument("--family-min-support", type=int, default=30)
    ap.add_argument("--family-jaccard", type=float, default=0.75)
    ap.add_argument("--family-name-sim", type=float, default=0.75)

    # fallbacks (optional)
    ap.add_argument("--fallback-local-stat", action="store_true",
                    help="If LLM omits stat_info, compute stat_info from predicate_histogram rarity (1 - freq).")
    ap.add_argument("--fallback-local-onto", action="store_true",
                    help="If LLM omits onto_info, estimate onto_info from type_rollup & generic_types.")

    # logging
    ap.add_argument("--log-dir", type=str, default=None, help="Base directory for logs. If omitted, uses logs/run-<timestamp>-<id>.")
    args = ap.parse_args()

    # Load triples
    if args.nt:
        nt_path = Path(args.nt)
        if nt_path.is_dir():
            if args.debug:
                print(f"[info] Reading .nt files recursively from directory: {nt_path}", file=sys.stderr)
            triples = load_triples_from_nt_dir(nt_path)
        else:
            triples = load_triples_from_nt(nt_path)
    elif args.json:
        with Path(args.json).open("r", encoding="utf-8") as f:
            data = json.load(f)
        triples = data.get("entity_description", [])
        if not triples:
            raise ValueError("JSON missing non-empty 'entity_description'")
    else:
        ap.error("Provide --nt or --json")
    
    dataset_id = infer_dataset_id(args)
    nt_filename = f"{dataset_id}_top{args.k}.nt"
    nt_filename_rank = f"{dataset_id}_rank_top{args.k}.nt"
    json_filename = f"{dataset_id}_top{args.k}.json"
    if not triples:
        raise ValueError("No triples loaded from the provided input.")

    # Learn generic types & predicate families from the entire corpus
    generic_types = learn_generic_types(
        triples,
        min_entities=args.min_entities_generic,
        df_frac_threshold=args.df_frac_threshold,
        min_df=args.min_df_generic
    )
    predicate_families = learn_predicate_families(
        triples,
        min_support=args.family_min_support,
        jaccard_threshold=args.family_jaccard,
        name_similarity_threshold=args.family_name_sim
    )

    # Build predicate -> family map (lowercased tails)
    family_of_pred: Dict[str,str] = {}
    for fam, preds in predicate_families.items():
        for p in preds:
            family_of_pred[p.lower()] = fam

    # Focus entity
    entity_id = args.entity_id or detect_primary_entity(triples)
    entity_triples = [t for t in triples if t["s"] == entity_id]
    if not entity_triples:
        raise ValueError("No triples found for the chosen entity.")

    entity_label = label_for_entity(entity_id, entity_triples)
    types_all = detect_types(entity_triples)
    type_rollup = filter_specific_types(types_all, generic_types, max_keep=5)
    pred_hist = build_predicate_histogram(entity_triples, top_n=args.topn_predicates)
    pred_examples = build_predicate_examples(entity_triples, per_pred=2)
    allowed = sorted({pred_tail(t["p"]) for t in entity_triples})

    triples_to_score = [
        {"id": t["id"], "p_tail": pred_tail(t["p"]), "o_preview": obj_preview(t["o"])}
        for t in entity_triples
    ]

    # LLM init
    gen, tok = build_llm_pipeline(args.llm_model_id, device=args.llm_device,
                                  dtype=args.llm_dtype, max_new_tokens=args.max_new_tokens)

    # Prepare logger
    short = hashlib.sha1((entity_id or "unknown").encode("utf-8")).hexdigest()[:8]
    run_meta = {
        "entity_id": entity_id,
        "entity_label": entity_label,
        "info_mode": args.info_mode,
        "k": args.k,
        "model_id": args.llm_model_id,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "beta": args.beta,
        "fallback_local_stat": bool(args.fallback_local_stat),
        "fallback_local_onto": bool(args.fallback_local_onto),
        "run_id": short,
    }
    logger = LogSink(args.log_dir, run_meta)

    # Build USER text once per batch
    def make_user_text(batch):
        base = {
            "INFO_MODE":         json.dumps(args.info_mode, ensure_ascii=False),
            "ENTITY_ID":         json.dumps(entity_id, ensure_ascii=False),
            "ENTITY_LABEL":      json.dumps(entity_label, ensure_ascii=False),
            "TYPE_ROLLUP":       json.dumps(type_rollup, ensure_ascii=False),
            "PRED_HISTOGRAM":    json.dumps(pred_hist, ensure_ascii=False),
            "PREDICATE_FAMILIES":json.dumps(predicate_families, ensure_ascii=False),
            "PREDICATE_EXAMPLES":json.dumps(pred_examples, ensure_ascii=False),
            "GENERIC_TYPES":     json.dumps(sorted(list(generic_types)), ensure_ascii=False),
            "TRIPLES_LIST":      json.dumps(batch, ensure_ascii=False),
            "ALLOWED_LIST":      json.dumps(allowed, ensure_ascii=False),
        }
        txt = USER_INFO_TEMPLATE
        for k, v in base.items():
            txt = txt.replace("{"+k+"}", v)
        return txt

    # Batch scoring
    batches = [triples_to_score[i:i+args.batch_size] for i in range(0, len(triples_to_score), args.batch_size)]
    id2score: Dict[str, Dict[str, float]] = {}

    for bidx, batch in enumerate(batches):
        user_text = make_user_text(batch)
        parsed = batch_score(
            gen, tok,
            system_prompt=SYSTEM_INFO,
            user_text=user_text,
            debug=args.debug,
            retries=2,
            delay=0.25,
            logger=logger,
            batch_idx=bidx,
            extra_meta={
                "entity_id": entity_id,
                "info_mode": args.info_mode,
                "k": args.k,
                "triples_in_batch": len(batch)
            }
        )

        # Map LLM outputs by triple id
        returned = {}
        if parsed:
            for it in parsed:
                tid = it.get("triple_id")
                if not isinstance(tid, str):
                    continue
                comps = it.get("info_components", {}) or {}
                s_info = it.get("stat_info", comps.get("stat_info", None))
                o_info = it.get("onto_info", comps.get("onto_info", None))
                inf    = it.get("informativeness", None)
                returned[tid] = {"stat_info": s_info, "onto_info": o_info, "informativeness": inf}

        # Consume scores for each triple in the batch
        for it in batch:
            tid = it["id"]
            # find predicate tail for fallbacks
            t_orig = next((tt for tt in entity_triples if tt["id"] == tid), None)
            ptail = pred_tail(t_orig["p"]) if t_orig else ""

            stat = returned.get(tid, {}).get("stat_info", None)
            onto = returned.get(tid, {}).get("onto_info", None)
            inf  = returned.get(tid, {}).get("informativeness", None)

            if stat is None and args.fallback_local_stat:
                stat = local_stat_for_pred(ptail, pred_hist)
            if onto is None and args.fallback_local_onto:
                onto = local_onto_for_pred(ptail, type_rollup, generic_types)

            stat = clamp01(stat) if stat is not None else 0.0
            onto = clamp01(onto) if onto is not None else 0.0

            if inf is None:
                inf = clamp01(args.beta * onto + (1.0 - args.beta) * stat)
            else:
                inf = clamp01(inf)

            id2score[tid] = {"stat_info": stat, "onto_info": onto, "informativeness": inf, "llm_informativeness": returned.get(tid,{}).get("informativeness", None)}

    # Build candidates, dedup near-duplicates by predicate+object tokens
    raw_candidates = []
    for t in entity_triples:
        sc = id2score.get(t["id"], {"stat_info":0.0, "onto_info":0.0, "informativeness":0.0, "llm_informativeness": None})
        raw_candidates.append({
            "triple_id": t["id"],
            "triple": {"s": t["s"], "p": t["p"], "o": t["o"]},
            "info_components": {
                "stat_info": round(sc["stat_info"], 3),
                "onto_info": round(sc["onto_info"], 3),
                "llm_informativeness": (round(sc["llm_informativeness"], 4) if isinstance(sc.get("llm_informativeness"), (int,float)) else None)
            },
            "informativeness": round(sc["informativeness"], 4),
            "notes": f"info-batch (mode={args.info_mode}, beta={args.beta})"
        })

    kept, seen = [], []
    for c in sorted(raw_candidates, key=lambda x: x["informativeness"], reverse=True):
        p_tail_c = pred_tail(c["triple"]["p"])
        o_prev = obj_preview(c["triple"]["o"]).lower()
        dup = any(jaccard_po_tokens(p_tail_c, o_prev, pp, oo) >= 0.8 for (pp, oo) in seen)
        if not dup:
            kept.append(c)
            seen.append((p_tail_c, o_prev))

    # Selection according to diversity policy
    if args.diversity == "none":
        selected = select_topk_no_diversity(kept, args.k)
    elif args.diversity == "family":
        fam_limits = parse_family_limit(args.family_limit.replace("==","="))
        family_of_pred_map: Dict[str,str] = {}
        for fam, preds in predicate_families.items():
            for p in preds:
                family_of_pred_map[p.lower()] = fam
        selected = select_summary_family(kept, args.k, family_of_pred_map, fam_limits)
    else:  # predicate
        selected = select_summary_predicate(kept, args.k)

    out = {
        "entity": entity_label,
        "entity_id": entity_id,
        "k": args.k,
        "diversity": args.diversity,
        "info_mode": args.info_mode,
        "candidates": kept,
        "selected_summary": selected,
        "meta": {
            "type_rollup": type_rollup,
            "predicate_histogram": pred_hist,
            "predicate_families": predicate_families,
            "learned_generic_types": sorted(list(generic_types)),
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "model_id": args.llm_model_id,
            "beta": args.beta,
            "fallback_local_stat": bool(args.fallback_local_stat),
            "fallback_local_onto": bool(args.fallback_local_onto)
        }
    }

    # Write outputs
    out_dir = ensure_out_dir(args.out, args.out_dir)
    write_json(out, args.out)
    if args.emit_csv:
        write_candidates_csv(kept, out_dir)
        write_selected_csv(selected, out_dir)
    if args.emit_md:
        write_markdown_report(entity_label=entity_label, info_mode=args.info_mode, meta=out["meta"],
                              candidates=kept, selected=selected, dest=out_dir)
    if args.emit_nt:
        write_selected_nt(selected, out_dir, filename=nt_filename)
        write_selected_nt(kept, out_dir, filename=nt_filename_rank)
    if args.pretty_console:
        print_pretty_console(out["entity"], selected, kept, pred_hist)

if __name__ == "__main__":
    main()
