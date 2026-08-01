# TODO.md

A running backlog of chapters or topics that shipped with a known limitation, an honestly-reported messy result, or something not verified as fully as it ideally would be. This is not a changelog — git history already covers what changed and why. This is specifically for the exceptions worth revisiting later. Entries are removed once actually revisited and resolved.

## Machine Learning

- **[Regularization](machine-learning/regularization.md)** — the unregularized degree-15 baseline in this chapter's example (200,698 test MSE) doesn't match the number `Bias vs Variance` published for the same setup (1,892), because solving via the normal equations directly is far more numerically unstable than `np.polyfit`'s method. The chapter names this honestly but doesn't explain the underlying numerical linear algebra reason in depth. Revisit if a deeper "why do these differ" treatment would help.

## LLMs

- **[Fine-Tuning](llms/fine-tuning.md)** — the verified example shows a pretrained model's zero-shot performance (83.3%) matching its fine-tuned performance on the same tiny dataset (83.3%), so fine-tuning's own marginal contribution isn't actually demonstrated, only the value of starting from pretrained weights. Revisit with a scenario where the target task differs enough from the pretraining distribution to leave real room for fine-tuning to improve on zero-shot, or with real LLM fine-tuning access.

## MLOps

- **[Kubernetes](mlops/kubernetes.md)** — YAML verified against current docs but never deployed against a live cluster; deliberately skipped since spinning up `minikube` would cost several minutes and real CPU/memory on this shared host. Revisit with an actual deployment if that cost becomes acceptable.
- **[CI/CD](mlops/ci-cd.md)** — YAML verified against current GitHub Actions docs but never run as a live workflow; deliberately not added to this repo's own `.github/workflows/`, since that would be infrastructure scope creep beyond writing the chapter. Revisit in a throwaway repo or sandbox if a live run is ever wanted.

## RAG

- **[Vector Databases](vector-databases/vector-databases.md)** — the example verifies the core ANN indexing mechanism directly (`hnswlib`), but never touches what actually makes something a *database*: metadata filtering, persistence, a real server product (Pinecone, Qdrant, Weaviate, Milvus). Revisit with an actual vector database product to demonstrate those parts, not just the underlying index algorithm.
