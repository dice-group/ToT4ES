# LMDB Semantic Encoding Challenge & Solution for ToT4ES

## Problem Analysis

### The Core Issue

Looking at LMDB entity description (entity 101 = film/12398):

```
<http://data.linkedmdb.org/resource/film/12398> <...movie/actor> <http://data.linkedmdb.org/resource/actor/29977> .
<http://data.linkedmdb.org/resource/director/82> <...foaf/0.1/made> <http://data.linkedmdb.org/resource/film/12398> .
```

**Problem**: When ToT4ES encounters `actor/29977` or `director/82`:
- These are just URIs with opaque numeric IDs
- No semantic label/type info is included in the current entity's description
- The semantic analyzer can only see the URI structure, not what these entities actually represent

### What We Need

To properly perform semantic analysis on LMDB:
1. Extract entity **type** from URI pattern (e.g., `actor/` → Actor, `film/` → Film)
2. Get entity **label** by looking up its description in other entity files
3. Build an **entity lookup cache** to quickly access semantic info

### Current Limitation

```python
# Current SemanticAnalyzer can parse:
parsed = {
    'subject': 'http://data.linkedmdb.org/resource/film/12398',
    'predicate': 'http://data.linkedmdb.org/resource/movie/actor',
    'object': 'http://data.linkedmdb.org/resource/actor/29977',  # ← Just a string!
    'is_literal': False,
    'predicate_local': 'actor'
}

# But cannot determine:
# - What type is actor/29977? (Actor? Organization?)
# - What is its label/name?
# - How common is this relation?
```

## Solution Architecture

### 1. Entity Knowledge Extractor

**File**: `scripts/tot_modules/knowledge_extractor.py`

Build a lookup system from LMDB directory:
```
URIPattern -> Type Mapping:
/film/NNN -> Film, Title (from rdfs:label)
/actor/NNN -> Actor, Name (from foaf:name)
/director/NNN -> Director, Name
/producer/NNN -> Producer
/writer/NNN -> Writer

Built from:
- URI path pattern recognition
- Scanning all entity description files
- Caching labels found via rdfs:label / foaf:name
```

### 2. LinkedMDB-Specific Semantic Analyzer

**File**: `scripts/tot_modules/lmdb_semantic_analyzer.py`

Extends SemanticAnalyzer with LMDB knowledge:
```python
class LMDBSemanticAnalyzer(SemanticAnalyzer):
    def __init__(self, all_triples, entity_cache):
        self.entity_cache = entity_cache  # Maps URI -> {'type': X, 'label': Y}
    
    def enrich_triple(self, triple_dict) -> Dict:
        """Add semantic metadata to parsed triple"""
        # Get object entity info
        obj_uri = triple_dict['object']
        if obj_uri in self.entity_cache:
            triple_dict['object_type'] = self.entity_cache[obj_uri]['type']
            triple_dict['object_label'] = self.entity_cache[obj_uri]['label']
```

### 3. LMDB Data Structures

```
LMDB dataset structure:
/lmdb_data/
  ├── 101-110, 141-175  (Film entities)
  │   └── {ID}_desc.nt  (Description files)
  └── Each contains:
      - rdf:type (Film)
      - rdfs:label (Movie title)
      - References to actor/{N}, director/{N}, writer/{N}

Key insight:
- If we encounter actor/12345 in film/101_desc.nt
- We might find actor/12345's details in actor entity file
- OR we can extract type from URI pattern + cached labels
```

## Implementation Steps

### Phase 1: URI Pattern Recognition (Quick Win)
```python
LMDB_URI_PATTERNS = {
    '/film/': 'Film',
    '/actor/': 'Actor', 
    '/director/': 'Director',
    '/writer/': 'Writer',
    '/producer/': 'Producer',
    '/character/': 'Character',
    '/genre/': 'Genre',
    '/performance/': 'Performance',
}
```

### Phase 2: Build Entity Cache from LMDB Directory
```python
def build_entity_cache(lmdb_data_path):
    """
    Scan all entity files in LMDB directory.
    For each entity, extract and cache:
    - URI
    - Type (from URI pattern or rdf:type)
    - Label (from rdfs:label, foaf:name, or dcterms:title)
    """
    cache = {}
    for entity_file in glob(f"{lmdb_data_path}/*/*.nt"):
        triples = load_triples_from_file(entity_file)
        for triple in triples:
            # Extract label predicates
            if 'label>' in triple or 'name>' in triple:
                uri = extract_subject_uri(triple)
                label = extract_literal_value(triple)
                cache[uri] = {
                    'type': extract_type_from_uri(uri),
                    'label': label
                }
    return cache
```

### Phase 3: Integrate with ToT4ES Semantic Prompts
```python
# In task_prompts.py or semantic_prompts.py
def make_lmdb_semantic_relatedness_prompt(
    entity_label: str,
    all_triples: List[str],
    entity_cache: Dict,  # NEW: Pass entity lookup cache
) -> Callable:
    analyzer = LMDBSemanticAnalyzer(all_triples, entity_cache)
    # Now can use semantic info about referenced entities
```

## Data Flow Diagram

```
LMDB Directory
    ↓
[Entity Cache Builder]
    ↓
{URI → Type, Label} Cache
    ↓
[LMDB Semantic Analyzer]
    ↓
Enriched Triple Analysis:
  - Subject type/label
  - Predicate type/frequency
  - Object type/label ✓ NEW
    ↓
[Enhanced ToT Prompts]
    ↓
Better triple selection
```

## Files to Create/Modify

### ✅ New Files (IMPLEMENTED):
1. ✅ `scripts/tot_modules/uri_utils.py` - URI pattern matching utilities (228 lines)
   - Extract entity types from URI patterns
   - Extract numeric IDs
   - Full URI categorization
   
2. ✅ `scripts/tot_modules/knowledge_extractor.py` - Entity cache builder (400+ lines)
   - Scan LMDB directory
   - Build entity cache with labels, types, predicates
   - Reference enrichment
   - Statistics and lookup methods
   
3. ✅ `scripts/tot_modules/lmdb_semantic_analyzer.py` - LMDB-specific analyzer (350+ lines)
   - Extends SemanticAnalyzer
   - Entity type/label lookup
   - LMDB predicate families
   - Enhanced scoring for informativeness/relatedness/diversity
   
4. ✅ `scripts/tot_modules/LMDB_INTEGRATION_GUIDE.md`
   - Usage examples for all components
   - Quick start guide
   - Performance notes
   
5. ✅ `test_lmdb_semantic_integration.py` - Comprehensive validation tests
   - Test all components
   - End-to-end integration test
   - Performance validation

### ⏳ To Modify (OPTIONAL - For Enhanced Integration):
1. `scripts/tot_entity_summarizer.py` - Optional: Add LMDB analyzer support
2. `scripts/tot_entity_summarizer_semantic.py` - Optional: Use LMDB analyzer for LMDB datasets
3. `scripts/tot_modules/semantic_prompts.py` - Optional: Pass entity cache to prompts

### 📚 Documentation Created:
1. ✅ `LMDB_SEMANTIC_ENCODING_INVESTIGATION.md` - Problem analysis (this file)
2. ✅ `LMDB_SEMANTIC_SOLUTION_IMPLEMENTED.md` - Complete implementation details

## Expected Benefits

1. **Semantic Understanding**: Can now understand what `actor/29977` means
2. **Better Relatedness Scoring**: Know entity types to judge centrality
3. **Improved Informativeness**: Understand predicate-object combinations
4. **Domain-Aware Analysis**: Recognize patterns like Film→Actor→{name} as informative

## Validation Strategy

```python
# Test on LMDB film/101:
entity_file = "lmdb_data/101/101_desc.nt"
cache = build_entity_cache("lmdb_data")

# Should find:
# film/12398 -> type: Film, label: "Men in White"
# actor/29977 -> type: Actor, label: (lookup from actor data)
# director/82 -> type: Director, label: (lookup)

print(cache["http://data.linkedmdb.org/resource/film/12398"])
# {'type': 'Film', 'label': 'Men in White'}

print(cache["http://data.linkedmdb.org/resource/actor/29977"])  
# {'type': 'Actor', 'label': '...'} if found in LMDB
# else: {'type': 'Actor', 'label': None} (type from pattern)
```

## Key Insight

The solution doesn't require external APIs or lookups:
- **Everything is in LMDB data itself**
- We just need to:
  1. Scan the directory to build entity index
  2. Extract type from URI patterns
  3. Cache labels from all entity files
  4. Use this cache during semantic analysis

This keeps ToT4ES **self-contained and offline**.
