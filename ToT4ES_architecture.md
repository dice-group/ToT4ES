# ToT4ES Pipeline Architecture (Task-Decomposed Entity Summarization)

This diagram describes the architecture and flow of the Task-Decomposed Tree-of-Thoughts Entity Summarization pipeline as implemented in this repository.

---

```mermaid
graph TD
    A[Input: Entity Description / KG (Triples)] --> B[Level 0: Empty Summary State]
    B --> C[Step 1: Expansion]
    C --> D1[Thought Generator: Relatedness]
    C --> D2[Thought Generator: Informativeness]
    C --> D3[Thought Generator: Diversity]
    D1 --> E[Candidate Pool (Triples)]
    D2 --> E
    D3 --> E
    E --> F[Evaluation Module\n(Thought Evaluation + Heuristic Scorer)]
    F --> G[Rank & Prune (Select Top N)]
    G --> H[Level 1 States (Summaries of len 1)]
    H --> I[Step 2: Expansion]
    I --> J1[Thought Generator: Relatedness]
    I --> J2[Thought Generator: Informativeness]
    I --> J3[Thought Generator: Diversity]
    J1 --> K[Candidate Pool (Triples)]
    J2 --> K
    J3 --> K
    K --> L[Evaluation Module\n(Thought Evaluation + Heuristic Scorer)]
    L --> M[Rank & Prune (Select Top N)]
    M --> N[Level 2 States (Summaries of len 2)]
    N --> O[Repeat Expansion for Levels 3-5]
    O --> P[Level 5 States (Final Summaries)]
    P --> Q[Select Best Summary]
    Q --> R[Final Output (Best Entity Summary)]
```

---

## Tree-of-Thoughts (ToT) Pattern Overview

The ToT pattern for entity summarization can be visualized as follows:

1. **At each expansion level (L=1,2,3...)**:
    - Multiple task-specific thought generators (experts for relatedness, informativeness, diversity) propose candidate triples to add to the current summary.
    - All candidates are pooled and evaluated.
    - The best candidates are selected to form partial summaries for the next level.
2. **This process repeats for each level**, building up the summary step by step.
3. **The final summary** is constructed from the best candidates at the last level.

### Mermaid Diagram Representation

```mermaid
flowchart TD
    subgraph L0[Partial Summary (L = 0)]
        S0[ ]
    end
    S0 --> TG1
    subgraph TG1[Thought Generators (Task-Specific Experts)]
        R1[Relatedness]
        I1[Informativeness]
        D1[Diversity]
    end
    TG1 --> C1[Candidate Pool]
    C1 --> E1[Evaluate & Prune]
    E1 --> L1[Partial Summary (L = 1)]

    L1 --> TG2
    subgraph TG2[Thought Generators (Task-Specific Experts)]
        R2[Relatedness]
        I2[Informativeness]
        D2[Diversity]
    end
    TG2 --> C2[Candidate Pool]
    C2 --> E2[Evaluate & Prune]
    E2 --> L2[Partial Summary (L = 2)]

    L2 --> TG3
    subgraph TG3[Thought Generators (Task-Specific Experts)]
        R3[Relatedness]
        I3[Informativeness]
        D3[Diversity]
    end
    TG3 --> C3[Candidate Pool]
    C3 --> E3[Evaluate & Prune]
    E3 --> L3[Partial Summary (L = 3)]

    L3 --> F[Final Summary]
```

---

This pattern matches the provided visual and the implemented code: at each level, the LLMs are queried for each task, their outputs are pooled and evaluated, and the best partial summaries are expanded until the final summary is produced.

**Legend:**
- **Thought Generators:** Propose candidate triples for expansion based on different criteria.
- **Candidate Pool:** Aggregates all proposed triples.
- **Evaluation Module:** Uses LLM and heuristics to score candidates.
- **Rank & Prune:** Selects top N candidates for the next level.
- **Repeat:** The process is repeated for each summary length up to the maximum (e.g., 5).
- **Final Output:** The best summary is selected from the final candidates.

---

This architecture matches both the code and the provided diagram, ensuring modular, interpretable, and extensible entity summarization.
