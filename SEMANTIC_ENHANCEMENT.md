# Semantic Enhancement for Entity Summarization

## Overview

This enhancement integrates **Description Logic (DL) principles** into ToT-based entity summarization to improve:
- ✅ **Informativeness**: Select specific, distinctive triples (not general)
- ✅ **Diversity**: Cover multiple aspects of the entity
- ✅ **Relatedness**: Focus on core, defining properties

## What's New?

### 1. **Semantic Analyzer** (`semantic_analyzer.py`)

Analyzes triples using DL-inspired heuristics:

```python
from tot_modules import SemanticAnalyzer

analyzer = SemanticAnalyzer(all_triples)

# Get informativeness scores (0-1)
info_scores = analyzer.get_informativeness_scores()
# → Rare predicates, functional properties score higher

# Get relatedness scores (0-1)
rel_scores = analyzer.get_relatedness_scores()
# → Defining predicates (rdf:type, label) score higher

# Get diversity hints
div_scores = analyzer.get_diversity_hints(selected_ids)
# → Different predicates/types score higher
```

#### **DL Principles Applied:**

| Principle | Implementation | Effect |
|-----------|---------------|--------|
| **Defining Properties** | Identifies rdf:type, rdfs:label, foaf:name | Prioritizes core entity identity |
| **Functional Properties** | Detects birthDate, birthPlace, isbn, etc. | Values unique values (high informativeness) |
| **Predicate Specificity** | Calculates inverse frequency | Rare predicates = more informative |
| **Type Diversity** | Balances literals vs entity links | Ensures varied information types |

### 2. **Enhanced Prompts** (`semantic_prompts.py`)

Prompts now include **semantic annotations**:

#### **Relatedness Prompt:**
```
Remaining candidates (with semantic hints):
1. <entity> rdf:type dbo:Scientist . [⭐DEFINING, CENTRAL]
2. <entity> dbo:birthDate "1879-03-14" . [🔑UNIQUE]
3. <entity> owl:sameAs <...> . (generic, no hint)
```

#### **Informativeness Prompt:**
```
Remaining candidates (with semantic hints):
5. <entity> dbo:almaMater <ETH_Zurich> . [🔑UNIQUE, ENTITY_LINK, HIGHLY_INFORMATIVE]
6. <entity> dbo:birthPlace <Ulm> . [🔑UNIQUE, ENTITY_LINK]
7. <entity> foaf:name "Albert Einstein" . (common, no hint)
```

#### **Diversity Prompt:**
```
Already selected: birthDate (literal), type (defining)

Remaining candidates (with semantic hints):
10. <entity> dbo:knownFor <...> . [🌈DIVERSE, ENTITY]
11. <entity> dbo:award <...> . [🌈DIVERSE, ENTITY]
```

### 3. **New Script** (`tot_entity_summarizer_semantic.py`)

Uses semantic enhancements throughout the pipeline.

## Usage

### **Basic Usage:**
```bash
python scripts/tot_entity_summarizer_semantic.py \
  --nt datasets/ESBM_benchmark_v1.2/dbpedia_data/1/1_desc.nt \
  --dataset dbpedia \
  --max-summary-len 5
```

### **With Semantic Analysis Display:**
```bash
python scripts/tot_entity_summarizer_semantic.py \
  --nt datasets/ESBM_benchmark_v1.2/dbpedia_data/1/1_desc.nt \
  --dataset dbpedia \
  --show-semantic-analysis
```

**Output:**
```
SEMANTIC ANALYSIS (DL-Inspired)
======================================================================

Entity Statistics:
  Total triples: 47
  Unique predicates: 23
  Defining triples (rdf:type, label): 3
  Functional properties (birthDate, etc.): 8
  Literals: 25
  Entity links: 22

Semantic Annotations:
  1. Info=0.35, Rel=0.90 | CORE | CENTRAL
     <entity> rdf:type dbo:Scientist .
  2. Info=0.75, Rel=0.45 | UNIQUE | HIGHLY_INFORMATIVE
     <entity> dbo:birthDate "1879-03-14"^^xsd:date .
  3. Info=0.82, Rel=0.30 | UNIQUE | HIGHLY_INFORMATIVE | ENTITY_LINK
     <entity> dbo:almaMater <http://dbpedia.org/resource/ETH_Zurich> .
```

## How It Improves Quality

### **1. Informativeness (Not General)**

**Before (no DL):**
```
- Might select common predicates everyone has
- No distinction between rare/common properties
- Equal weight to metadata and substantive facts
```

**After (with DL):**
```python
# Semantic scoring penalizes general predicates:
if predicate in general_predicates:  # sameAs, seeAlso, wikiPageID
    score -= 0.3

# Rewards rare, specific predicates:
specificity = 1.0 - (predicate_frequency)
score += specificity * 0.5

# Rewards functional properties:
if predicate in functional_predicates:  # birthDate, isbn
    score += 0.3
```

**Result:** Selects **distinctive, specific** triples like:
- `dbo:almaMater <ETH_Zurich>` (specific institution)
- `dbo:nobelPrizeIn "Physics"` (rare achievement)

Instead of generic ones like:
- `owl:sameAs <wikidata:...>` (just linking)
- `dbo:wikiPageID "736"` (metadata)

### **2. Diversity (Multiple Aspects)**

**Before:**
```
- Might select similar predicates (birthDate, deathDate, foundingDate)
- Could focus on one aspect (all biographical OR all professional)
```

**After:**
```python
# Tracks selected predicate types:
selected_preds = ['birthDate', 'birthPlace']

# Penalizes duplicate predicate types:
if new_predicate in selected_preds:
    diversity_score -= 0.5

# Rewards type variety:
if not has_entity_link_yet and is_entity_link:
    diversity_score += 0.25
```

**Result:** Balanced coverage:
- Biographical: `birthDate`, `birthPlace`
- Professional: `occupation`, `almaMater`
- Achievement: `award`, `knownFor`
- Relational: `influenced`, `colleague`

### **3. Relatedness (Core Properties)**

**Before:**
```
- Treats all predicates equally
- Might prioritize rare predicates over defining ones
```

**After:**
```python
# High priority for defining predicates:
if predicate in defining_predicates:  # rdf:type, rdfs:label
    relatedness_score += 0.6

# Common predicates are central:
frequency = predicate_count / total
relatedness_score += frequency * 0.4
```

**Result:** Always includes **core identity** triples:
- `rdf:type dbo:Scientist` (what IS it?)
- `rdfs:label "Albert Einstein"@en` (how to refer to it?)
- Common for this type: `dbo:field "Physics"` (typical for scientists)

## Comparison

### **Standard ToT:**
```
Selected summary (may be imbalanced):
1. <entity> rdf:type dbo:Scientist .
2. <entity> dbo:birthDate "1879-03-14" .
3. <entity> dbo:deathDate "1955-04-18" .
4. <entity> dbo:birthPlace <Ulm> .
5. <entity> dbo:deathPlace <Princeton> .

→ All biographical, lacks diversity
```

### **Semantic ToT:**
```
Selected summary (DL-guided):
1. <entity> rdf:type dbo:Scientist . [CORE]
2. <entity> dbo:almaMater <ETH_Zurich> . [INFORMATIVE, ENTITY_LINK]
3. <entity> dbo:knownFor <Theory_of_Relativity> . [INFORMATIVE, DIVERSE]
4. <entity> dbo:award <Nobel_Prize_in_Physics> . [INFORMATIVE, DIVERSE]
5. <entity> dbo:birthPlace <Ulm> . [UNIQUE, DIVERSE]

→ Balanced: identity + education + achievement + location
→ Informative: specific entities, not just dates
→ Diverse: multiple aspects covered
```

## Integration Points

The semantic enhancements guide **three key stages**:

### **1. Thought Generation** (Prompt Level)
```python
# LLM sees semantic hints in prompts
"Select from:
 5. <entity> dbo:almaMater <ETH> . [🔑UNIQUE, HIGHLY_INFORMATIVE]
 6. <entity> owl:sameAs <...> . (no hint = less appealing)"
```

### **2. State Evaluation** (Scoring Level)
```python
# Heuristic can use semantic scores
def enhanced_heuristic(state):
    selected_ids = parse_state(state)
    
    # Weight by semantic quality
    info_score = mean([analyzer.get_informativeness_scores()[i] 
                       for i in selected_ids])
    rel_score = mean([analyzer.get_relatedness_scores()[i] 
                      for i in selected_ids])
    
    return 0.4 * rel_score + 0.4 * info_score + 0.2 * diversity
```

### **3. Post-Analysis** (Explainability)
```python
# Show WHY each triple was selected
"Triple 2: Informativeness=0.85, Relatedness=0.45
 Semantic: UNIQUE | HIGHLY_INFORMATIVE | ENTITY_LINK"
```

## Configuration

All features work with existing parameters:

```bash
# Use semantic enhancements with DFS
python scripts/tot_entity_summarizer_semantic.py \
  --nt data.nt \
  --dataset dbpedia \
  --search-algorithm dfs

# Custom models per task
python scripts/tot_entity_summarizer_semantic.py \
  --nt data.nt \
  --dataset dbpedia \
  --model-relatedness meta-llama/Llama-3.2-1B-Instruct \
  --model-informativeness mistralai/Mistral-7B-Instruct-v0.2

# Show detailed analysis
python scripts/tot_entity_summarizer_semantic.py \
  --nt data.nt \
  --dataset dbpedia \
  --show-semantic-analysis \
  --no-verbose
```

## Files Modified/Created

**New Files:**
- `tot_modules/semantic_analyzer.py` - DL-based triple analysis
- `tot_modules/semantic_prompts.py` - Enhanced prompts with semantic hints
- `scripts/tot_entity_summarizer_semantic.py` - Main script using semantic enhancements

**Modified Files:**
- `tot_modules/__init__.py` - Export semantic components

## Benefits Summary

| Aspect | Without DL | With DL (Semantic) |
|--------|-----------|-------------------|
| **Informativeness** | May select common predicates | Prioritizes rare, distinctive properties |
| **Diversity** | May focus on one aspect | Balances multiple aspects (bio, prof, etc.) |
| **Relatedness** | Equal weight to all | Core properties (type, label) get priority |
| **Guidance** | Generic prompts | Semantic hints guide LLM better |
| **Explainability** | Score only | Semantic annotations explain WHY |

## Recommendation

**Use Semantic ToT when:**
- Quality > speed (slightly more computation for analysis)
- You want **informative** summaries (not just frequent predicates)
- You need **diverse** coverage (multiple entity aspects)
- You want **explainability** (semantic annotations)

**Stick with Standard ToT when:**
- Speed is critical
- Dataset has very simple/uniform structure
- You're just prototyping

---

## Example Command

```bash
# Run semantic-enhanced ToT on DBpedia entity
python scripts/tot_entity_summarizer_semantic.py \
  --nt datasets/ESBM_benchmark_v1.2/dbpedia_data/1/1_desc.nt \
  --dataset dbpedia \
  --max-summary-len 5 \
  --show-semantic-analysis
```

This will produce summaries that are:
- ✅ More **informative** (specific, not general)
- ✅ More **diverse** (multiple aspects)
- ✅ More **related** (core properties included)

All guided by Description Logic principles! 🎯
