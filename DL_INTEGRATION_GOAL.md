# Goal of Implementing DL in Entity Summarization

## **Primary Goal**
Improve entity summary **quality** by producing summaries that are:
1. **More Informative** (specific, not generic)
2. **More Diverse** (covers multiple aspects)  
3. **More Related** (includes core properties)

---

## **The Problem Without DL**

**Standard ToT approach treats all triples equally:**
- No distinction between `rdf:type` (defining) and `owl:sameAs` (generic link)
- No awareness that `birthDate` is more informative than `wikiPageID`
- May select redundant information (birthDate, deathDate, foundingDate = all dates)

**Result:** Summaries might be:
- ❌ Too generic (metadata instead of substantive facts)
- ❌ Imbalanced (all biographical, no professional achievements)
- ❌ Missing core identity (no `rdf:type` or `label`)

---

## **How DL Principles Solve This**

### **1. Informativeness (Specific > General)**

**DL Concept:** Predicates have different **specificity levels**

**Implementation:**
```python
# Rare predicates = more informative
dbo:nobelPrizeIn → appears in 0.1% of entities → HIGH informativeness
dbo:birthDate → appears in 60% of entities → MEDIUM informativeness
owl:sameAs → appears in 95% of entities → LOW informativeness

# Functional properties = unique values = informative
birthDate: one value per person → informative
almaMater: specific institution → informative
wikiPageID: just metadata → not informative
```

**Result:** Selects distinctive facts like `nobelPrizeIn` over generic links like `sameAs`

---

### **2. Diversity (Multiple Aspects)**

**DL Concept:** Entities have multiple **facets/roles** in ontology

**Implementation:**
```python
# Track predicate types
Selected: [birthDate (temporal), birthPlace (spatial)]

# Prioritize different types
AVOID: deathDate (another temporal) → diversity score LOW
PREFER: occupation (professional) → diversity score HIGH
PREFER: knownFor (achievement) → diversity score HIGH

# Balance literals vs entity links
If selected = [all literals], prefer entity links next
If selected = [all entity links], prefer literals next
```

**Result:** Balanced coverage of biographical + professional + achievements + relationships

---

### **3. Relatedness (Core > Peripheral)**

**DL Concept:** Some properties **define** entity identity (TBox axioms)

**Implementation:**
```python
# Defining predicates (from ontology structure)
rdf:type → defines WHAT the entity IS → HIGH relatedness
rdfs:label → how to NAME it → HIGH relatedness
dbo:field → common for Scientists → MEDIUM relatedness
dbo:award → distinctive but not defining → LOWER relatedness

# Common predicates for entity type = central
For Scientist: field, almaMater, knownFor → central
For Scientist: birthPlace, spouse → less central
```

**Result:** Always includes core identity triples (`type`, `label`) that define the entity

---

## **DL Integration Strategy**

### **Where DL Is Applied:**

1. **Semantic Analysis** (Pre-processing)
   - Analyze all triples once
   - Classify predicates: defining, functional, general
   - Calculate specificity scores (inverse frequency)

2. **Enhanced Prompts** (Thought Generation)
   - Add semantic hints to guide LLM
   - `[⭐DEFINING]` → prioritize for relatedness
   - `[🔑UNIQUE]` → prioritize for informativeness
   - `[🌈DIVERSE]` → prioritize for diversity

3. **Scoring** (Evaluation)
   - Weight triples by semantic quality
   - Defining predicates get relatedness bonus
   - Rare predicates get informativeness bonus
   - Different types get diversity bonus

---

## **Concrete Example**

### **Entity: Albert Einstein**

**Without DL (may select):**
```
1. owl:sameAs <wikidata:Q937>        ← generic link (not informative)
2. dbo:birthDate "1879-03-14"        ← biographical
3. dbo:deathDate "1955-04-18"        ← biographical (redundant with #2)
4. dbo:birthPlace <Ulm>              ← biographical  
5. dbo:wikiPageID "736"              ← metadata (not informative)

→ Too biographical, includes metadata, lacks achievements
```

**With DL (semantic-guided):**
```
1. rdf:type dbo:Scientist            ← DEFINING (DL priority)
2. dbo:almaMater <ETH_Zurich>        ← UNIQUE, ENTITY_LINK (informative)
3. dbo:knownFor <Relativity>         ← RARE, ACHIEVEMENT (informative + diverse)
4. dbo:award <Nobel_Prize>           ← RARE, ACHIEVEMENT (informative + diverse)
5. dbo:birthPlace <Ulm>              ← UNIQUE, SPATIAL (diverse from others)

→ Balanced: identity + education + achievements + location
→ Informative: specific entities, not metadata
→ Diverse: multiple aspects (professional + biographical)
```

---

## **Expected Quality Improvements**

| Metric | Without DL | With DL |
|--------|-----------|---------|
| **Informativeness** | May include generic metadata | Prioritizes distinctive facts |
| **Diversity** | May cluster on one aspect | Covers multiple facets |
| **Relatedness** | May miss core properties | Always includes type/label |
| **Redundancy** | May select similar triples | Avoids duplicate information |
| **Explainability** | Just scores | Semantic annotations explain WHY |

---

## **Implementation Components**

### **1. Semantic Analyzer** (`tot_modules/semantic_analyzer.py`)
```python
class SemanticAnalyzer:
    def get_predicate_specificity() -> Dict[str, float]:
        """Rare predicates = high score (informativeness)"""
        
    def get_triple_categories() -> Dict[int, Dict[str, bool]]:
        """Classify: defining, functional, general, literal, entity_link"""
        
    def get_informativeness_scores() -> Dict[int, float]:
        """Based on: specificity + functional + entity_link"""
        
    def get_relatedness_scores() -> Dict[int, float]:
        """Based on: defining predicates + common patterns"""
        
    def get_diversity_hints() -> Dict[int, float]:
        """Based on: different predicates + type balance"""
```

### **2. Enhanced Prompts** (`tot_modules/semantic_prompts.py`)
```python
def make_semantic_relatedness_prompt():
    """Adds [⭐DEFINING, CENTRAL] hints to guide LLM"""

def make_semantic_informativeness_prompt():
    """Adds [🔑UNIQUE, HIGHLY_INFORMATIVE] hints"""

def make_semantic_diversity_prompt():
    """Adds [🌈DIVERSE] hints based on selected triples"""
```

### **3. Main Script** (`tot_entity_summarizer_semantic.py`)
- Uses semantic analysis throughout pipeline
- Shows statistics (defining count, functional count, etc.)
- Annotates selected triples with DL insights

---

## **Usage**

```bash
# Run with semantic enhancement
python scripts/tot_entity_summarizer_semantic.py \
  --nt datasets/ESBM_benchmark_v1.2/dbpedia_data/1/1_desc.nt \
  --dataset dbpedia \
  --max-summary-len 5 \
  --show-semantic-analysis

# Output includes:
# - Entity statistics (defining, functional, literals, entity links)
# - Semantic annotations for each triple
# - Informativeness and relatedness scores
```

---

## **Summary**

**DL integration uses ontology principles to guide triple selection:**

| DL Principle | Purpose | Implementation |
|--------------|---------|----------------|
| **Defining Properties** | Core identity | Prioritize rdf:type, rdfs:label |
| **Functional Properties** | Unique values | Bonus for birthDate, isbn, etc. |
| **Predicate Specificity** | Informativeness | Inverse frequency scoring |
| **Type Diversity** | Multiple aspects | Balance literals vs entity links |
| **Common Patterns** | Relatedness | Frequent predicates for entity type |

**Result:** Summaries that are more informative (specific facts), more diverse (multiple aspects), and more related (core identity) compared to treating all triples equally.

---

## **Key Insight**

**Without DL:** "Select 5 best triples" → may pick metadata, generic links, imbalanced

**With DL:** "Select 5 best triples where:
- At least 1 DEFINING (type/label)
- Prefer RARE predicates (informative)
- Avoid SIMILAR predicates (diverse)
- Include SPECIFIC values (not metadata)"

→ Structured quality criteria based on semantic understanding 🎯
