# ToT4ES: Tree of Thought for Entity Summarization (Think, Branch, Summarize)

This repository contains the implementation of ToT4ES, a research paper that submitted to EKAW 2026 conference (Research track). ToT4ES is an unsupervised method of extractive entity summarization relies on LLMs through Tree of Tought strategy that decomposes into three complementary semantic objectives: relatedness, informativeness, and coverage, to select entity summaries.

<p align="center">
<img src="images/tot4es-architecture.png" width="75%">
</p>

## ⚙️ Installation
To run the ToT4ES framework, you need to install the following packages:

```bash
python 3.10+
torch
```

1. Create and activate a Conda environment:
```bash
conda create --name tot4es-env python=3.10
conda activate tot4es-env
```
2. Download the project
```bash
git clone https://github.com/dice-group/ToT4ES.git

# Navigate to ToT4ES directory
cd ToT4ES
```

2. Install required packages:
```bash
pip install torch
pip install -r requirements.txt
```

> ⚠️ **Important Note:** Ensure that all dependencies are correctly installed.

## 🚀 Quickstart
A minimal quickstart to run a small demo using the repository's test script.

1. Create and activate a virtual environment (venv):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. (Optional) If you plan to use Hugging Face models that require authentication, login:

```bash
huggingface-cli login
```

4. Run a quick dataset check/demo:

```bash
python test.py
```

## ▶ Running ToT4ES

Follow these steps to run the baseline ToT4ES pipeline 
- Run a single-entity example (quick test):

```bash
python3 scripts/tot_entity_summarizer_task_decomposed.py \
	--nt datasets/ESBM_benchmark_v1.2/dbpedia_data/1/1_desc.nt \
	--dataset dbpedia \
	--max-summary-len 5
```

- Run the full baseline processing (iterates over dataset entities):

```bash
cd scripts
./run_tot4es.sh
```

