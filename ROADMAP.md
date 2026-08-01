# ROADMAP.md

# AI Engineer Learning Roadmap

This document defines **what exists** in the repository and the recommended learning order.

Every topic listed here should eventually become a dedicated Markdown chapter.

---

# Phase 1 — Foundations

## Machine Learning

- [x] What is Machine Learning
- [x] Types of Machine Learning
- [x] ML Workflow
- [ ] Bias vs Variance
- [ ] Feature Engineering

## Statistics

- [ ] Mean
- [ ] Median
- [ ] Mode
- [ ] Variance
- [ ] Standard Deviation
- [ ] Probability
- [ ] Bayes Theorem

---

# Phase 2 — Data Processing

## Python for ML

- [ ] NumPy
- [ ] Pandas
- [ ] Polars

## Data Visualization

- [ ] Matplotlib
- [ ] Plotly

---

# Phase 3 — NLP

## NLP Fundamentals

- [ ] Tokenization
- [ ] Text Cleaning
- [ ] Embeddings
- [ ] Similarity

---

# Phase 4 — LLMs

## Foundations

- [ ] Transformers
- [ ] Attention
- [ ] Prompt Engineering

---

# Phase 5 — RAG

## Retrieval Augmented Generation

- [ ] What is RAG
- [ ] Chunking
- [ ] Retrieval
- [ ] Reranking
- [ ] Evaluation

---

# Phase 6 — Agents

- [ ] Agent Architecture
- [ ] LangGraph
- [ ] Tool Calling
- [ ] Multi-Agent Systems

---

# Phase 7 — Production AI

- [ ] FastAPI
- [ ] Docker
- [ ] Kubernetes
- [ ] CI/CD
- [ ] Observability

---

# Placement Rules

These rules apply when a proposed topic does not map cleanly to exactly one existing folder listed in `README.md`.

## No Home

If a topic does not fit any existing folder:

- Do not invent a new folder or category.
- Explain why it doesn't fit the current structure.
- Ask the user to decide whether to add a new folder or reshape an existing one.

## Multiple Homes

If a topic fits more than one existing folder:

- Write the full explanation once, in the folder representing its primary mechanism or origin, not its downstream use.
- Reference it from the other relevant folders using relative Markdown links.
- Do not duplicate the explanation across folders.
- If it's unclear which folder is primary, explain the tradeoff and ask instead of guessing.

---

# Notes

- This file defines repository structure.
- Do not place educational content here.
- Each topic should correspond to a dedicated chapter.