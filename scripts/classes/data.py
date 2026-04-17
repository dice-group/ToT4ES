import os
import re
import glob
import numpy as np
import datetime
from typing import Optional, List, Dict, Tuple, Any

# ------------------------------------------------------------
# Triple normalization & robust lookup
# ------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_TRIPLE_RE = re.compile(r'^(?P<s><[^>]+>)\s+(?P<p><[^>]+>)\s+(?P<o>.+?)(?:\s*\.)?$')

def _collapse_ws(s: str) -> str:
    return _WS_RE.sub(" ", s.strip())

def _split_triple(line: str):
    """Parse a single N-Triples line into (s, p, o_raw) without trailing dot."""
    m = _TRIPLE_RE.match(line.strip())
    if not m:
        return None, None, None
    return m.group("s"), m.group("p"), m.group("o")

def _literal_variants(o_raw: str):
    """
    For literal objects, generate variants tolerant to:
      - presence/absence of language tag (@en, @en-US, etc.)
      - presence/absence of datatype (^^<...>)
    For IRIs, return as-is.
    """
    x = o_raw.strip()
    if not x.startswith('"'):
        return [x]

    # find end of the quoted literal (handle escapes)
    core_end = None
    esc = False
    for i, ch in enumerate(x[1:], start=1):
        if ch == '\\' and not esc:
            esc = True
            continue
        if ch == '"' and not esc:
            core_end = i
            break
        esc = False
    if core_end is None:
        return [x]  # malformed literal; best effort

    literal_core = x[:core_end + 1]
    rest = x[core_end + 1:].strip()

    out = []
    out.append(x)  # as-is

    has_lang = rest.startswith("@")
    has_dtype = "^^<" in rest

    # strip lang
    if has_lang:
        m = re.match(r"^@[a-zA-Z]+(?:-[a-zA-Z0-9]+)?(.*)$", rest)
        rest_wo_lang = m.group(1).strip() if m else rest
        out.append(_collapse_ws(f"{literal_core} {rest_wo_lang}".strip()))
        out.append(literal_core)  # just the core

    # strip datatype
    if has_dtype:
        before, _, after = rest.partition("^^<")
        # drop ^^<...>
        rest_wo_dtype = before.strip()
        out.append(_collapse_ws(f"{literal_core} {rest_wo_dtype}".strip()))
        out.append(literal_core)

    # fully stripped core
    out.append(literal_core)

    # dedup preserve order
    seen, dedup = set(), []
    for v in out:
        v2 = v.strip()
        if v2 not in seen:
            seen.add(v2)
            dedup.append(v2)
    return dedup

def _variants_for_triple(line: str):
    """
    Generate normalized key variants for a triple line to make dictionary
    building and lookups tolerant to formatting differences.
    """
    s, p, o = _split_triple(line)
    if not s:
        base = line.strip()
        if base.endswith("."):
            base = base[:-1].strip()
        return [_collapse_ws(base)]
    o_vars = _literal_variants(o)
    variants = []
    for ov in o_vars:
        variants.append(_collapse_ws(f"{s} {p} {ov}"))
    base = line.strip()
    if base.endswith("."):
        base = base[:-1]
    variants.append(_collapse_ws(base))
    # dedup
    seen, out = set(), []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out

def _lookup_encoded(triples_dict: dict, raw_line: str):
    """Return encoded id from triples_dict using robust variants, or None."""
    for key in _variants_for_triple(raw_line):
        if key in triples_dict:
            return triples_dict[key]
    return None

# ------------------------------------------------------------
# Output file discovery (supports ES_OUT_ROOT override)
# ------------------------------------------------------------

def _candidate_roots(db_path: str) -> list:
    """
    Priority:
      1) ES_OUT_ROOT (env), also ES_OUT_ROOT/<dataset>
      2) db_path (original)
    """
    roots = []
    env_root = os.environ.get("ES_OUT_ROOT", "").strip()
    if env_root:
        ds_name = os.path.basename(db_path.rstrip(os.sep))
        roots.append(os.path.join(env_root, ds_name))
        roots.append(env_root)
    roots.append(db_path)
    return [os.path.abspath(r) for r in roots]

def _find_first(paths: list) -> Optional[str]:
    for p in paths:
        if os.path.isfile(p):
            return p
    return None

def _find_rank_file(db_path: str, num: int, top_n: int) -> str:
    nums_to_try = [num] + ([num - 1] if num > 0 else [])
    roots = _candidate_roots(db_path)

    primary = []
    for r in roots:
        for n in nums_to_try:
            primary.append(os.path.join(r, f"{n}", f"{n}_rank_top{top_n}.nt"))
            primary.append(os.path.join(r, f"{n}", f"{n}_rank.nt"))
    p = _find_first(primary)
    if p:
        return p

    fallbacks = []
    for r in roots:
        for n in nums_to_try:
            for ext in ("ttl", "txt"):
                fallbacks.append(os.path.join(r, f"{n}", f"{n}_rank_top{top_n}.{ext}"))
                fallbacks.append(os.path.join(r, f"{n}", f"{n}_rank.{ext}"))
    p = _find_first(fallbacks)
    if p:
        return p

    for r in roots:
        for n in nums_to_try:
            base = os.path.join(r, f"{n}")
            for cand in glob.glob(os.path.join(base, "*rank*")):
                if os.path.isfile(cand):
                    return cand

    raise FileNotFoundError(
        f"Rank file not found for num={num}, top_n={top_n}. "
        f"Searched roots: {', '.join(roots)}"
    )

def _find_topk_file(db_path: str, num: int, top_n: int) -> str:
    nums_to_try = [num] + ([num - 1] if num > 0 else [])
    roots = _candidate_roots(db_path)

    primary = []
    for r in roots:
        for n in nums_to_try:
            for ext in ("nt", "ttl", "txt"):
                primary.append(os.path.join(r, f"{n}", f"{n}_top{top_n}.{ext}"))
    p = _find_first(primary)
    if p:
        return p

    for r in roots:
        for n in nums_to_try:
            base = os.path.join(r, f"{n}")
            for cand in glob.glob(os.path.join(base, f"*top{top_n}.*")):
                if os.path.isfile(cand):
                    return cand

    raise FileNotFoundError(
        f"Top-k file not found for num={num}, top_n={top_n}. "
        f"Searched roots: {', '.join(roots)}"
    )

# ------------------------------------------------------------
# Your original API (patched to use the helpers above)
# ------------------------------------------------------------

# Functions for data loading and preprocessing
def load_dglke(ds_name):
    """Load pre-trained graph embeddings"""
    input_entity_dict = f"{ds_name}-esbm/entities.tsv"
    input_relation_dict = f"{ds_name}-esbm/relations.tsv"
    entity2ix = build_dictionary(input_entity_dict)
    pred2ix = build_dictionary(input_relation_dict)
    entity2vec = np.load(f'{ds_name}-kge-model/ComplEx_{ds_name}/{ds_name}_ComplEx_entity.npy', mmap_mode='r')
    pred2vec = np.load(f'{ds_name}-kge-model/ComplEx_{ds_name}/{ds_name}_ComplEx_relation.npy', mmap_mode='r')
    # entity2vec = np.load(f'{ds_name}-kge-model/DistMult_{ds_name}/{ds_name}_DistMult_entity.npy', mmap_mode='r')
    # pred2vec = np.load(f'{ds_name}-kge-model/DistMult_{ds_name}/{ds_name}_DistMult_relation.npy', mmap_mode='r')
    return entity2vec, pred2vec, entity2ix, pred2ix

def build_dictionary(input_dict):
    with open(input_dict, "r", encoding="utf8") as f:
        content = f.readlines()
    idx2dict = {}
    for items in content:
        items = items.rstrip("\n").split("\t")
        idx = int(items[0])
        dict_value = items[1]
        idx2dict[dict_value] = idx
    return idx2dict

def format_triples(triples):
    formatted_triples = []
    for triple in triples:
        h, r, t = triple
        head = h.split('/')[-1]
        try:
            clean_relation = r.split('/')[-1]
        except Exception as e:
            print(triple)
            print(e)
            break
        clean_relation = re.sub(r'.*#', '', clean_relation)
        relation = clean_relation
        t = str(t)
        if t.startswith('http://') or t.startswith('https://'):
            tail = t.split('/')[-1]
        else:
            clean_literal = re.sub(r'\^\^<http.*', '', t)
            clean_literal = clean_literal.replace('"', '')
            clean_literal = clean_literal.replace('@e', '')  # keep your original behavior
            tail = clean_literal
        if len(tail) > 0:
            input_formatted = f"{head}[SEP]{relation}[SEP]{tail}"
            formatted_triples.append(input_formatted)
        else:
            input_formatted = f"{head}[SEP]{relation}"
            formatted_triples.append(input_formatted)
    return formatted_triples

def writer(db_dir, directory, eid, top_or_rank, topk, rank_list):
    "Write triples to file"
    with open(os.path.join(db_dir, f"{eid}", f"{eid}_desc.nt"), encoding="utf8") as fin:
        with open(os.path.join(directory, f"{eid}_{top_or_rank}{topk}.nt"), "w", encoding="utf8") as fout:
            triples = [triple for _, triple in enumerate(fin)]
            for rank in rank_list:
                fout.write(triples[rank])

def get_rank_triples(db_path, num, top_n, triples_dict):
    triples = []
    encoded_triples = []
    # resilient discovery first, then fall back to legacy layout
    try:
        filename = _find_rank_file(db_path, num, top_n)
    except FileNotFoundError:
        filename = os.path.join(db_path, f"{num}", f"{num}_rank_top{top_n}.nt")
        if not os.path.exists(filename):
            filename = os.path.join(db_path, f"{num}", f"{num}_rank.nt")

    with open(filename, encoding="utf8") as reader:
        for _, triple in enumerate(reader):
            triple = triple.replace("\n", "").strip()
            if not triple:
                continue
            triples.append(triple)
            enc = _lookup_encoded(triples_dict, triple)
            if enc is None:
                continue
            encoded_triples.append(enc)
    return triples, encoded_triples

def get_topk_triples(db_path, num, top_n, triples_dict):
    triples = []
    encoded_triples = []
    # resilient discovery
    try:
        filename = _find_topk_file(db_path, num, top_n)
    except FileNotFoundError:
        filename = os.path.join(db_path, f"{num}", f"{num}_top{top_n}.nt")

    with open(filename, encoding="utf8") as reader:
        for _, triple in enumerate(reader):
            triple = triple.replace("\n", "").strip()
            if not triple:
                continue
            triples.append(triple)
            enc = _lookup_encoded(triples_dict, triple)
            if enc is None:
                continue
            encoded_triples.append(enc)
    return triples, encoded_triples

def get_all_data(db_path, num, top_n, file_n):
    triples_dict = {}
    triple_tuples = []

    # read all triples for entity <num>
    desc_path = os.path.join(db_path, f"{num}", f"{num}_desc.nt")
    with open(desc_path, encoding="utf8") as reader:
        for _, triple in enumerate(reader):
            triple = triple.strip()
            if not triple:
                continue
            triple_tuples.append(triple)
            variants = _variants_for_triple(triple)
            if not variants:
                continue
            # assign id to canonical (first) variant if unseen
            if variants[0] not in triples_dict:
                triples_dict[variants[0]] = len(triples_dict)
            cur_id = triples_dict[variants[0]]
            # map all variants to same id
            for v in variants[1:]:
                triples_dict.setdefault(v, cur_id)

    gold_list = []
    ds_name = db_path.split("/")[-1].split("_")[0]

    # faces dataset: adjust file_n to actual number of gold files
    if ds_name == "faces":
        gold_files = glob.glob(os.path.join(db_path, f"{num}", f"{num}_gold_top{top_n}_*"))
        if len(gold_files) != file_n:
            file_n = len(gold_files)

    # read gold summaries
    for i in range(file_n):
        gold_path = os.path.join(db_path, f"{num}", f"{num}_gold_top{top_n}_{i}.nt")
        with open(gold_path, encoding="utf8") as reader:
            n_list = []
            for _, triple in enumerate(reader):
                triple = triple.strip()
                if not triple:
                    continue
                # resolve using variants; if unseen in desc, add now
                variants = _variants_for_triple(triple)
                enc = None
                for v in variants:
                    if v in triples_dict:
                        enc = triples_dict[v]
                        break
                if enc is None:
                    # add new id for previously unseen gold triple
                    triples_dict[variants[0]] = len(triples_dict)
                    enc = triples_dict[variants[0]]
                    for v in variants[1:]:
                        triples_dict.setdefault(v, enc)
                n_list.append(enc)
            gold_list.append(n_list)

    return gold_list, triples_dict, triple_tuples

def format_time(elapsed):
    """Takes a time in seconds and returns a string hh:mm:ss"""
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))
