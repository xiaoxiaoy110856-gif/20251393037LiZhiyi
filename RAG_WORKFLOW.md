# RAG Workflow for Trajectory and Reinforcement Learning

This project now supports a more stable local RAG pipeline for trajectory and reinforcement-learning questions.

## What changed

- documents are split into topic-aware chunks instead of indexing whole files as single blocks
- each chunk stores metadata:
  - `title`
  - `path`
  - `section`
  - `topics`
  - `doc_type`
- retrieval now mixes:
  - vector recall
  - lexical boosts
  - research-focus boosts

## Main scripts

- `scripts/build_rag_corpus.py`
  - rebuilds the parsed corpus JSON with chunk metadata
- `scripts/build_rag_index.py`
  - rebuilds the parsed corpus and the vector index
- `scripts/eval_rag.py`
  - evaluates retrieval hit rate and optional answer grounding
- `scripts/ingest_local_kb.py`
  - legacy convenience entrypoint; still works

## Recommended local workflow

### 1. Rebuild the corpus

```powershell
python scripts/build_rag_corpus.py
```

### 2. Rebuild the index

```powershell
python scripts/build_rag_index.py
```

### 3. Evaluate retrieval quality

```powershell
python scripts/eval_rag.py --data "training_data/rag_eval.jsonl"
```

### 4. Evaluate retrieval + answer generation

```powershell
python scripts/eval_rag.py --data "training_data/rag_eval.jsonl" --answer
```

## Knowledge organization

The corpus uses these main topic tags:

- `trajectory_compression`
- `trajectory_prediction`
- `path_planning`
- `reinforcement_learning`
- `ppo`
- `dqn`
- `sac`
- `offline_rl`
- `reward_design`
- `trajectory_similarity`
- `experiment_design`
- `project_code`

If you add new papers or notes, try to keep filenames and section headings descriptive. The chunk builder uses those fields during retrieval reranking.

## Paper enrichment

The file below contains curated online paper notes that were added to strengthen the RAG seed set:

- `kb/raw/trajectory_papers/curated_rl_trajectory_rag_notes.md`

You can keep extending that file with:

- title
- source / venue
- link
- short relevance summary
- topic tags

This is a lightweight way to enrich the knowledge base before downloading every PDF.
