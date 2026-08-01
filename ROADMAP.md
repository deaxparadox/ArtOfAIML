# ROADMAP.md

# AI Engineer Learning Roadmap

This document defines **what exists** in the repository and the recommended learning order.

Every topic listed here should eventually become a dedicated Markdown chapter.

---

# Phase 1 — Foundations

## Machine Learning

- [x] [What is Machine Learning](machine-learning/what-is-machine-learning.md)
- [x] [Types of Machine Learning](machine-learning/types-of-machine-learning.md)
- [x] [Supervised Learning](machine-learning/supervised-learning.md)
- [x] [Unsupervised Learning](machine-learning/unsupervised-learning.md)
- [x] [Semi-Supervised Learning](machine-learning/semi-supervised-learning.md)
- [x] [Reinforcement Learning](machine-learning/reinforcement-learning.md)
- [x] [Self-Supervised Learning](machine-learning/self-supervised-learning.md)
- [x] [ML Workflow](machine-learning/ml-workflow.md)
- [x] [Bias vs Variance](machine-learning/bias-vs-variance.md)
- [x] [Regularization](machine-learning/regularization.md)
- [ ] Cross-Validation
- [ ] Ensemble Methods
- [x] [Feature Engineering](machine-learning/feature-engineering.md)

## Statistics

- [x] [Mean](statistics/mean.md)
- [x] [Median](statistics/median.md)
- [x] [Mode](statistics/mode.md)
- [x] [Variance](statistics/variance.md)
- [x] [Standard Deviation](statistics/standard-deviation.md)
- [ ] Normal Distribution
- [x] [Probability](statistics/probability.md)
- [x] [Bayes Theorem](statistics/bayes-theorem.md)
- [ ] Correlation vs Causation
- [ ] Hypothesis Testing / Confidence Intervals

---

# Phase 2 — Data Processing

## Python for ML

- [x] [NumPy](python-for-ml/numpy.md)
- [x] [Pandas](python-for-ml/pandas.md)
- [x] [Polars](python-for-ml/polars.md)

## Data Visualization

- [x] [Matplotlib](data-visualization/matplotlib.md)
- [x] [Plotly](data-visualization/plotly.md)

---

# Phase 3 — NLP

## NLP Fundamentals

- [x] [Tokenization](nlp/tokenization.md)
- [x] [Text Cleaning](nlp/text-cleaning.md)
- [x] [Embeddings](embeddings/embeddings.md) (lives in `embeddings/` — see Placement Rules)
- [x] [Similarity](embeddings/similarity.md) (lives in `embeddings/` — see Placement Rules)

---

# Phase 4 — LLMs

## Foundations

- [x] [Transformers](llms/transformers.md)
- [x] [Attention](llms/attention.md)
- [x] [Prompt Engineering](llms/prompt-engineering.md)

---

# Phase 5 — RAG

## Retrieval Augmented Generation

- [x] [What is RAG](rag/what-is-rag.md)
- [x] [Chunking](rag/chunking.md)
- [x] [Retrieval](rag/retrieval.md)
- [x] [Reranking](rag/reranking.md)
- [x] [Evaluation](rag/evaluation.md)

---

# Phase 6 — Agents

- [x] [Agent Architecture](agents/agent-architecture.md)
- [x] [LangGraph](agents/langgraph.md)
- [x] [Tool Calling](agents/tool-calling.md)
- [x] [Multi-Agent Systems](agents/multi-agent-systems.md)

---

# Phase 7 — Production AI

- [x] [FastAPI](mlops/fastapi.md) (lives in `mlops/` — see Placement Rules)
- [x] [Docker](mlops/docker.md) (lives in `mlops/` — see Placement Rules)
- [x] [Kubernetes](mlops/kubernetes.md) (lives in `mlops/` — see Placement Rules)
- [x] [CI/CD](mlops/ci-cd.md) (lives in `mlops/` — see Placement Rules)
- [x] [Observability](mlops/observability.md) (lives in `mlops/` — see Placement Rules)

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