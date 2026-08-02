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
- [x] [Cross-Validation](machine-learning/cross-validation.md)
- [x] [Ensemble Methods](machine-learning/ensemble-methods.md)
- [x] [Feature Engineering](machine-learning/feature-engineering.md)
- [x] [SVM](machine-learning/svm.md)
- [x] [Naive Bayes](machine-learning/naive-bayes.md)
- [x] [PCA](machine-learning/pca.md)

## Statistics

- [x] [Mean](statistics/mean.md)
- [x] [Median](statistics/median.md)
- [x] [Mode](statistics/mode.md)
- [x] [Variance](statistics/variance.md)
- [x] [Standard Deviation](statistics/standard-deviation.md)
- [x] [Normal Distribution](statistics/normal-distribution.md)
- [x] [Probability](statistics/probability.md)
- [x] [Bayes Theorem](statistics/bayes-theorem.md)
- [x] [Correlation vs Causation](statistics/correlation-vs-causation.md)
- [x] [Hypothesis Testing / Confidence Intervals](statistics/hypothesis-testing-confidence-intervals.md)

---

# Phase 2 — Data Processing

## Python for ML

- [x] [NumPy](python-for-ml/numpy.md)
- [x] [Pandas](python-for-ml/pandas.md)
- [x] [Polars](python-for-ml/polars.md)

## Data Visualization

- [x] [Matplotlib](data-visualization/matplotlib.md)
- [x] [Plotly](data-visualization/plotly.md)
- [x] [Seaborn](data-visualization/seaborn.md)

---

# Phase 3 — NLP

## NLP Fundamentals

- [x] [Tokenization](nlp/tokenization.md)
- [x] [Text Cleaning](nlp/text-cleaning.md)
- [x] [TF-IDF / Bag-of-Words](nlp/tfidf-bag-of-words.md)
- [x] [Embeddings](embeddings/embeddings.md) (lives in `embeddings/` — see Placement Rules)
- [x] [Similarity](embeddings/similarity.md) (lives in `embeddings/` — see Placement Rules)

---

# Phase 4 — LLMs

## Foundations

- [x] [Transformers](llms/transformers.md)
- [x] [Attention](llms/attention.md)
- [x] [Context Window / Context Length](llms/context-window.md)
- [x] [Prompt Engineering](llms/prompt-engineering.md)
- [x] [Fine-Tuning](llms/fine-tuning.md)
- [x] [LLM Evaluation / Benchmarks](llms/llm-evaluation-benchmarks.md)

---

# Phase 5 — RAG

## Retrieval Augmented Generation

- [x] [What is RAG](rag/what-is-rag.md)
- [x] [Chunking](rag/chunking.md)
- [x] [Query Rewriting / Query Expansion](rag/query-rewriting.md)
- [x] [Retrieval](rag/retrieval.md)
- [x] [Vector Databases](vector-databases/vector-databases.md)
- [x] [Reranking](rag/reranking.md)
- [x] [Evaluation](rag/evaluation.md)
- [x] [Agentic RAG](rag/agentic-rag.md)

---

# Phase 6 — Agents

- [x] [Agent Architecture](agents/agent-architecture.md)
- [x] [Agent Memory](agents/agent-memory.md)
- [x] [LangGraph](agents/langgraph.md)
- [x] [Tool Calling](agents/tool-calling.md)
- [x] [Guardrails](agents/guardrails.md)
- [x] [Human-in-the-Loop](agents/human-in-the-loop.md)
- [x] [Multi-Agent Systems](agents/multi-agent-systems.md)
- [x] [Agent Evaluation](agents/agent-evaluation.md)

---

# Phase 7 — Production AI

- [x] [FastAPI](mlops/fastapi.md) (lives in `mlops/` — see Placement Rules)
- [x] [LLM Serving](mlops/llm-serving.md)
- [x] [Feature Store](mlops/feature-store.md)
- [x] [Docker](mlops/docker.md) (lives in `mlops/` — see Placement Rules)
- [x] [Kubernetes](mlops/kubernetes.md) (lives in `mlops/` — see Placement Rules)
- [x] [Canary Deployment / Model Rollout](mlops/canary-deployment.md)
- [x] [Model Registry](mlops/model-registry.md)
- [x] [CI/CD](mlops/ci-cd.md) (lives in `mlops/` — see Placement Rules)
- [x] [Observability](mlops/observability.md) (lives in `mlops/` — see Placement Rules)
- [x] [Drift Monitoring](mlops/drift-monitoring.md)

---

# Phase 8 — Deep Learning Fundamentals

Added after all 7 original phases were expanded — a genuine gap identified and confirmed with the user, since [Transformers](llms/transformers.md) had no neural-network groundwork beneath it. Appended as its own phase rather than inserted earlier, per the no-renumbering rule in `CLAUDE.md`.

- [x] [Neural Networks](deep-learning/neural-networks.md)
- [x] [Backpropagation](deep-learning/backpropagation.md)
- [x] [Activation Functions](deep-learning/activation-functions.md)
- [x] [Optimizers](deep-learning/optimizers.md)
- [ ] CNNs
- [ ] RNNs / LSTMs
- [ ] Regularization for Deep Nets

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