#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entity Diversity/Coverage Summarizer (LLM-derived semantic/textual features)
with Greedy Coverage Selection + Predicate/Family limits + Rich Outputs + NT export + Batch Logs
-----------------------------------------------------------------------------------------------
- Input: N-Triples (.nt) file OR a directory of .nt files (recursive), or JSON {"entity_description":[...]}.
- Learns generic types (IDF-style) + predicate families (seeds + subject-set Jaccard + name similarity).
- Summarizes context: type_rollup (drop generic types), predicate_histogram, predicate_examples, predicate_families.

No informativeness scores. Selection maximizes coverage/diversity using:
  • Discrete coverage: new predicate, predicate family, role_tag
  • Textual coverage: new object shingles (character/word n-grams)
  • Semantic coverage: new semantic_keys (LLM keywords) and low redundancy (Jaccard penalty)

Greedy objective (marginal gain):
  gain = w_fam*(new family) + w_role*(new role) + w_pred*(new predicate)
       + w_keys*(new semantic keys fraction) + w_text*(new text shingles fraction)
  penalty = redundancy_lambda * max_jaccard_to_selected (semantic_keys ∪ text_keys)
  select the candidate with highest (gain - penalty) iteratively until k

Selection limits:
  --diversity none       : pure greedy coverage (no per-predicate/family limits)
  --diversity predicate  : greedy coverage but at most one per predicate
  --diversity family     : greedy coverage with per-family limits (see --family-limit)

Outputs:
  --out (JSON), --emit-csv, --emit-md, --emit-nt, --pretty-console
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

SYSTEM_DIVERSE = (
    "You are an entity summarization assistant optimizing coverage/diversity ONLY. "
    "Work strictly with the provided inputs. Return valid JSON ONLY — no prose, no backticks.\n\n"
    "For each triple, derive:\n"
    "- role_tag: one tag from {identity,label,type,identifier,homepage,location,timeEvent,affiliation,relation,media,other} "
    "(pick the closest; use 'other' if uncertain)\n"
    "- semantic_keys: 3-8 lowercase keywords capturing the unique meaning of this triple for THIS entity "
    "(e.g., place names, years, ids, specific roles). No spaces inside a key; use hyphens or underscores.\n"
    "- text_keys: 3-8 lowercase surface tokens from the object preview (normalize, drop stopwords if obvious)\n\n"
    "No external knowledge. Keys must be derived from the provided strings.\n"
    "Output format strictly:\n"
    "'[{\"triple_id\":\"<id>\",\"role_tag\":\"<tag>\",\"semantic_keys\":[\"...\"],\"text_keys\":[\"...\"]}]'\n"
)

USER_DIVERSE_TEMPLATE = (
    "Entity:\n"
    "- id: {ENTITY_ID}\n"
    "- preferred_label: {ENTITY_LABEL}\n"
    "- type_rollup: {TYPE_ROLLUP}\n\n"
    "Context summary:\n"
    "- predicate_histogram: {PRED_HISTOGRAM}\n"
    "- predicate_families: {PREDICATE_FAMILIES}\n"
    "- predicate_examples: {PREDICATE_EXAMPLES}\n"
    "- learned_generic_types: {GENERIC_TYPES}\n\n"
    "Triples_to_tag:\n"
    "{TRIPLES_LIST}\n\n"
    "Constraints:\n"
    "- allowed_relations: {ALLOWED_LIST}\n\n"
    "Return JSON array with keys 'triple_id','role_tag','semantic_keys','text_keys'."
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

# ============================= Coverage helpers ============================

def shingle_tokens(s: str) -> Set[str]:
    # simple tokenization to coverage shingles (word unigrams + 3-5 char ngrams)
    tokens = [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t]
    chars = set()
    joined = "".join(tokens)
    for n in (3,4,5):
        for i in range(max(0, len(joined)-n+1)):
            chars.add(joined[i:i+n])
    return set(tokens) | chars

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    return len(a & b) / float(len(a | b))

# =============================== Selection (Greedy) ========================

def greedy_coverage_select(
    candidates: List[Dict[str, Any]],
    k: int,
    family_of_pred: Dict[str,str],
    per_family_limit: Dict[str,int],
    diversity_mode: str,
    w_fam: float,
    w_role: float,
    w_pred: float,
    w_keys: float,
    w_text: float,
    redundancy_lambda: float
) -> List[Dict[str, Any]]:
    selected = []
    seen_fams: Set[str] = set()
    seen_preds: Set[str] = set()
    seen_roles: Set[str] = set()
    covered_keys: Set[str] = set()
    covered_text: Set[str] = set()
    fam_counts: Dict[str,int] = defaultdict(int)

    # precompute feature sets
    for c in candidates:
        pt = pred_tail(c["triple"]["p"]).lower()
        fam = family_of_pred.get(pt, f"pred:{pt}")
        c["_pt"] = pt
        c["_fam"] = fam
        c["_role"] = c["coverage"].get("role_tag","other")
        c["_sem_keys"] = set(c["coverage"].get("semantic_keys", []))
        # derive textual keys unioned with shingles from preview
        text_keys = set(c["coverage"].get("text_keys", []))
        text_keys |= shingle_tokens(obj_preview(c["triple"]["o"]))
        c["_text_keys"] = text_keys

    def fam_limit(f: str) -> int:
        return per_family_limit.get(f, per_family_limit.get("*", 1))

    used_ids: Set[str] = set()
    while len(selected) < k and len(selected) < len(candidates):
        best = None
        best_gain = -1e9

        for c in candidates:
            if c["triple_id"] in used_ids:
                continue

            # predicate/family diversity constraints if requested
            if diversity_mode == "predicate" and c["_pt"] in seen_preds:
                continue
            if diversity_mode == "family" and fam_counts[c["_fam"]] >= fam_limit(c["_fam"]):
                continue

            new_fam = 1.0 if c["_fam"] not in seen_fams else 0.0
            new_role = 1.0 if c["_role"] not in seen_roles else 0.0
            new_pred = 1.0 if c["_pt"] not in seen_preds else 0.0

            new_sem_keys = len(c["_sem_keys"] - covered_keys)
            sem_den = max(1, len(c["_sem_keys"]))
            frac_sem_new = new_sem_keys / sem_den

            new_text_keys = len(c["_text_keys"] - covered_text)
            txt_den = max(1, len(c["_text_keys"]))
            frac_text_new = new_text_keys / txt_den

            # redundancy penalty against current selection
            max_j_sem = 0.0
            for s in selected:
                js = jaccard(c["_sem_keys"], s["_sem_keys"])
                jt = jaccard(c["_text_keys"], s["_text_keys"])
                max_j_sem = max(max_j_sem, max(js, jt))

            gain = (
                w_fam * new_fam +
                w_role * new_role +
                w_pred * new_pred +
                w_keys * frac_sem_new +
                w_text * frac_text_new
            ) - redundancy_lambda * max_j_sem

            if gain > best_gain:
                best_gain = gain
                best = (c, gain)

        if best is None:
            break

        c, gain = best
        selected.append({**c, "marginal_gain": round(gain, 4)})
        used_ids.add(c["triple_id"])

        # update coverage sets
        seen_fams.add(c["_fam"])
        seen_roles.add(c["_role"])
        seen_preds.add(c["_pt"])
        covered_keys |= c["_sem_keys"]
        covered_text |= c["_text_keys"]
        fam_counts[c["_fam"]] += 1

    # add order and slim output
    return [
        {
            "order": i+1,
            "triple_id": x["triple_id"],
            "triple": x["triple"],
            "marginal_gain": x["marginal_gain"],
            "coverage": x["coverage"]
        }
        for i, x in enumerate(selected[:k])
    ]

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
        w.writerow(["triple_id","s","p","o","role_tag","semantic_keys","text_keys"])
        for c in cands:
            s = c["triple"]["s"]; p = c["triple"]["p"]; o = c["triple"]["o"]
            cov = c.get("coverage", {})
            w.writerow([c["triple_id"], s, p, o,
                        cov.get("role_tag",""),
                        " ".join(cov.get("semantic_keys", [])),
                        " ".join(cov.get("text_keys", []))])
    return fp

def write_selected_csv(sel: List[Dict[str, Any]], dest: Path):
    fp = dest / "selected.csv"
    with fp.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order","triple_id","s","p","o","role_tag","marginal_gain"])
        for x in sel:
            cov = x.get("coverage", {})
            w.writerow([x["order"], x["triple_id"], x["triple"]["s"], x["triple"]["p"], x["triple"]["o"],
                        cov.get("role_tag",""), x.get("marginal_gain","")])
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

def write_markdown_report(entity_label: str, meta: dict, candidates: List[Dict[str,Any]], selected: List[Dict[str,Any]], dest: Path):
    fp = dest / "report.md"
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    def md_escape(s: str) -> str:
        return str(s).replace("|","\\|")
    with fp.open("w", encoding="utf-8") as f:
        f.write(f"# Entity summary report\n\n")
        f.write(f"- entity: {entity_label}\n- generated: {ts}\n\n")
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
        f.write("## selected summary (greedy coverage)\n\n")
        f.write("| # | predicate | object (preview) | role | marginal_gain |\n|---|---|---|---|---:|\n")
        for x in selected:
            p_tail = pred_tail(x["triple"]["p"])
            o_prev = obj_preview(x["triple"]["o"])
            role = x.get("coverage",{}).get("role_tag","")
            mg   = x.get("marginal_gain", 0.0)
            f.write(f"| {x['order']} | {md_escape(p_tail)} | {md_escape(o_prev)} | {md_escape(role)} | {mg:.4f} |\n")
        f.write("\n## top candidates (up to 30)\n\n")
        f.write("| rank | triple_id | predicate | object (preview) | role | semantic_keys | text_keys |\n|---|---|---|---|---|---|---|\n")
        for i, c in enumerate(candidates[:30], 1):
            p_tail = pred_tail(c["triple"]["p"])
            o_prev = obj_preview(c["triple"]["o"])
            cov = c.get("coverage", {})
            f.write(f"| {i} | {md_escape(c['triple_id'])} | {md_escape(p_tail)} | {md_escape(o_prev)} | "
                    f"{md_escape(cov.get('role_tag',''))} | {md_escape(' '.join(cov.get('semantic_keys', [])))} | "
                    f"{md_escape(' '.join(cov.get('text_keys', [])))} |\n")
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
    print("Selected summary (k={}): Greedy coverage".format(len(selected)))
    print("{:>2}  {:<18}  {:<55}  {:<10} {:>9}".format("#","predicate","object (preview)","role","Δgain"))
    line()
    for x in selected:
        p = pred_tail(x["triple"]["p"])[:18]
        o = obj_preview(x["triple"]["o"], 55)
        role = (x.get("coverage",{})).get("role_tag","")[:10]
        mg = x.get("marginal_gain", 0.0)
        print("{:>2}  {:<18}  {:<55}  {:<10} {:>9.4f}".format(x["order"], p, o, role, mg))
    line()
    print("Top candidates (showing first 15):")
    print("{:>3} {:<8} {:<18} {:<46} {:<10} {:<18}".format("rk","id","predicate","object (preview)","role","sem_keys"))
    line()
    for i, c in enumerate(candidates[:15], 1):
        p = pred_tail(c["triple"]["p"])[:18]
        o = obj_preview(c["triple"]["o"], 46)
        role = c.get("coverage",{}).get("role_tag","")[:10]
        semk = " ".join(c.get("coverage",{}).get("semantic_keys", [])[:3])[:18]
        print("{:>3} {:<8} {:<18} {:<46} {:<10} {:<18}".format(i, c["triple_id"], p, o, role, semk))
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

def _json_sanitize(obj):
    """Recursively remove private keys (starting with '_') and convert sets to lists."""
    if isinstance(obj, dict):
        clean = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.startswith("_"):
                continue  # drop private/internal fields
            clean[k] = _json_sanitize(v)
        return clean
    elif isinstance(obj, list):
        return [_json_sanitize(x) for x in obj]
    elif isinstance(obj, set):
        # make a stable list for reproducibility
        try:
            return sorted(list(obj))
        except Exception:
            return list(obj)
    else:
        return obj


def main():
    ap = argparse.ArgumentParser(description="Entity Diversity/Coverage Summarizer with greedy selection, semantic/textual features, and batch logs (LLM-derived).")
    ap.add_argument("--nt", type=str, help="Path to an N-Triples file OR a directory containing .nt files (recursively)")
    ap.add_argument("--json", type=str, help="Path to JSON with key 'entity_description'")
    ap.add_argument("--entity-id", type=str, default=None, help="Override entity subject IRI")
    ap.add_argument("--k", type=int, default=5, help="Number of triples to select")

    # Diversity mode: controls per-predicate/family limits only; selection is always greedy coverage
    ap.add_argument("--diversity", choices=["none","predicate","family"], default="predicate",
                    help="Selection limits: none=unconstrained; predicate=one per predicate; family=per-family limits.")
    ap.add_argument("--family-limit", type=str, default="*==1",
                    help='Only used when --diversity family. Comma-separated limits, e.g., "type=2,*=1"')

    # LLM
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--llm-model-id", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--llm-device", type=str, default="auto")
    ap.add_argument("--llm-dtype", type=str, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=1024)

    # Output
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

    # Greedy coverage weights
    ap.add_argument("--w-fam", type=float, default=0.6, help="Weight for covering a new predicate family")
    ap.add_argument("--w-role", type=float, default=0.5, help="Weight for covering a new role_tag")
    ap.add_argument("--w-pred", type=float, default=0.3, help="Weight for covering a new predicate")
    ap.add_argument("--w-keys", type=float, default=0.8, help="Weight for adding new semantic_keys (fraction)")
    ap.add_argument("--w-text", type=float, default=0.4, help="Weight for adding new text shingles (fraction)")
    ap.add_argument("--redundancy-lambda", type=float, default=0.7, help="Penalty * max Jaccard(similarity) to selected")

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
    pred_hist = build_predicate_histogram(entity_triples, top_n=20)
    pred_examples = build_predicate_examples(entity_triples, per_pred=2)
    allowed = sorted({pred_tail(t["p"]) for t in entity_triples})

    triples_to_tag = [
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
        "k": args.k,
        "model_id": args.llm_model_id,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "w_fam": args.w_fam,
        "w_role": args.w_role,
        "w_pred": args.w_pred,
        "w_keys": args.w_keys,
        "w_text": args.w_text,
        "redundancy_lambda": args.redundancy_lambda,
        "diversity_mode": args.diversity,
        "run_id": short,
    }
    logger = LogSink(args.log_dir, run_meta)

    # Build USER text once per batch
    def make_user_text(batch):
        base = {
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
        txt = USER_DIVERSE_TEMPLATE
        for k, v in base.items():
            txt = txt.replace("{"+k+"}", v)
        return txt

    # Batch LLM tagging
    batches = [triples_to_tag[i:i+args.batch_size] for i in range(0, len(triples_to_tag), args.batch_size)]
    id2cov: Dict[str, Dict[str, Any]] = {}

    for bidx, batch in enumerate(batches):
        user_text = make_user_text(batch)
        raw_text = None
        parsed = None
        for attempt in range(3):
            try:
                prompt = apply_chat_template(tok, SYSTEM_DIVERSE, user_text)
                out = gen(prompt, num_return_sequences=1)
                raw_text = extract_text(out)
                parsed = parse_or_repair_json_array(raw_text)
                logger.log_batch(
                    batch_idx=bidx,
                    attempt=attempt,
                    system_prompt=SYSTEM_DIVERSE,
                    user_prompt=user_text,
                    raw_text=raw_text,
                    parsed_obj=parsed,
                    extra_meta={
                        "entity_id": entity_id,
                        "k": args.k,
                        "triples_in_batch": len(batch)
                    }
                )
                if parsed is None or not isinstance(parsed, list):
                    raise ValueError("LLM did not return a JSON array")
                break
            except Exception as e:
                if args.debug:
                    print(f"[batch] attempt {attempt} error: {e}", file=sys.stderr)
                    print("RAW_LLM_OUT_START", raw_text if raw_text is not None else "", "RAW_LLM_OUT_END", file=sys.stderr)
                time.sleep(0.25)

        if parsed:
            for it in parsed:
                tid = it.get("triple_id")
                if not isinstance(tid, str):
                    continue
                role = it.get("role_tag","other")
                semk = it.get("semantic_keys", []) or []
                textk = it.get("text_keys", []) or []
                # normalize tokens
                semk = [norm_token(x) for x in semk if norm_token(x)]
                textk = [norm_token(x) for x in textk if norm_token(x)]
                id2cov[tid] = {"role_tag": role, "semantic_keys": semk, "text_keys": textk}

        # fill missing with minimal coverage (will still get shingles later)
        for it in batch:
            tid = it["id"]
            id2cov.setdefault(tid, {"role_tag":"other","semantic_keys": [], "text_keys": []})

    # Build candidates and dedup near-duplicates by predicate+object tokens
    raw_candidates = []
    for t in entity_triples:
        cov = id2cov.get(t["id"], {"role_tag":"other","semantic_keys":[],"text_keys":[]})
        raw_candidates.append({
            "triple_id": t["id"],
            "triple": {"s": t["s"], "p": t["p"], "o": t["o"]},
            "coverage": {
                "role_tag": cov["role_tag"],
                "semantic_keys": cov["semantic_keys"],
                "text_keys": cov["text_keys"],
            },
            "notes": "coverage-batch"
        })

    def jaccard_po_tokens(a_pred: str, a_obj: str, b_pred: str, b_obj: str) -> float:
        A = set(re.findall(r'\w+', (a_pred + " " + a_obj).lower()))
        B = set(re.findall(r'\w+', (b_pred + " " + b_obj).lower()))
        if not A and not B: return 1.0
        return len(A & B) / max(1, len(A | B))

    kept, seen = [], []
    for c in raw_candidates:
        p_tail_c = pred_tail(c["triple"]["p"])
        o_prev = obj_preview(c["triple"]["o"]).lower()
        dup = any(jaccard_po_tokens(p_tail_c, o_prev, pp, oo) >= 0.8 for (pp, oo) in seen)
        if not dup:
            kept.append(c)
            seen.append((p_tail_c, o_prev))

    # Selection according to greedy coverage with optional predicate/family limits
    fam_limits = parse_family_limit(args.family_limit.replace("==","="))
    family_of_pred_map: Dict[str,str] = {}
    for fam, preds in predicate_families.items():
        for p in preds:
            family_of_pred_map[p.lower()] = fam

    selected = greedy_coverage_select(
        candidates=kept,
        k=args.k,
        family_of_pred=family_of_pred_map,
        per_family_limit=fam_limits,
        diversity_mode=args.diversity,
        w_fam=args.w_fam, w_role=args.w_role, w_pred=args.w_pred,
        w_keys=args.w_keys, w_text=args.w_text,
        redundancy_lambda=args.redundancy_lambda
    )

    out = {
        "entity": entity_label,
        "entity_id": entity_id,
        "k": args.k,
        "diversity_mode": args.diversity,
        "candidates": _json_sanitize(kept),
        "selected_summary": _json_sanitize(selected),
        "meta": {
            "type_rollup": type_rollup,
            "predicate_histogram": pred_hist,
            "predicate_families": predicate_families,
            "learned_generic_types": sorted(list(generic_types)),
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "model_id": args.llm_model_id,
            "weights": {
                "w_fam": args.w_fam, "w_role": args.w_role, "w_pred": args.w_pred,
                "w_keys": args.w_keys, "w_text": args.w_text,
                "redundancy_lambda": args.redundancy_lambda
            }
        }
    }

    # Write outputs
    out_dir = ensure_out_dir(args.out, args.out_dir)
    write_json(out, args.out)
    if args.emit_csv:
        write_candidates_csv(kept, out_dir)
        write_selected_csv(selected, out_dir)
    if args.emit_md:
        write_markdown_report(entity_label=entity_label, meta=out["meta"],
                              candidates=kept, selected=selected, dest=out_dir)
    if args.emit_nt:
        write_selected_nt(selected, out_dir, filename=nt_filename)
        # keep a rank file of all kept for inspection
        write_selected_nt(
            [{"triple": c["triple"]} for c in kept],
            out_dir, filename=nt_filename_rank
        )
    if args.pretty_console:
        print_pretty_console(out["entity"], selected, kept, pred_hist)

if __name__ == "__main__":
    main()
