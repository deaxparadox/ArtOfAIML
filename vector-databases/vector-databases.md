# Vector Databases

See [Retrieval](../rag/retrieval.md) for the approximate nearest neighbor search this chapter's infrastructure is built around.

## What is it?

A vector database is a database purpose-built for storing embeddings and searching them efficiently by similarity, rather than by exact match — the infrastructure that makes the retrieval step in a real RAG system fast at scale.

[Retrieval](../rag/retrieval.md) established why brute-force comparison against every stored vector doesn't scale, and that production systems use approximate nearest neighbor (ANN) search instead. A vector database is where that index actually lives: a specialized storage and query engine built around exactly this "find the closest vectors" operation, the way a traditional database is built around exact lookups and range queries.

## Why does it exist?

[Retrieval](../rag/retrieval.md) established the core problem — given a query embedding, find the closest stored embeddings among potentially millions, without comparing against every single one. A vector database exists to package that ANN search, along with the practical concerns any real system needs — storing metadata alongside vectors, updating and deleting entries, filtering by both similarity and traditional criteria like a date or category, persisting to disk, scaling across machines — into one purpose-built system, rather than requiring every RAG application to hand-build its own indexing and storage layer from scratch.

**A general-purpose database's vector extension vs. a dedicated vector database, as a real decision.** Many traditional databases (Postgres via `pgvector`, for example) now offer vector search as an add-on feature — a reasonable choice when the vectors live alongside a lot of other structured, relational data already managed there, and vector search is a secondary need. A dedicated vector database is the right choice when vector search *is* the primary workload, at a scale or performance requirement a general-purpose database's add-on wasn't built to handle as its main job.

## How does it work?

Each vector is stored alongside an ID and often metadata — the original text chunk, a source document, a category. An approximate nearest neighbor index is built over the stored vectors, commonly a graph-based algorithm like HNSW, trading a small, tunable chance of missing the true best match for dramatically faster search — exactly the trade-off [Retrieval](../rag/retrieval.md) already introduced. A query, itself an embedding, is compared against that index rather than the raw stored vectors, retrieving the top-k approximate nearest neighbors quickly, with metadata filters able to narrow the search further.

## Example

A real approximate nearest neighbor index (HNSW, via `hnswlib`), compared directly against brute-force exact search over 50,000 vectors:

```python
import numpy as np
import time
import hnswlib

rng = np.random.default_rng(0)
data = rng.normal(size=(50000, 64)).astype("float32")
query = rng.normal(size=(1, 64)).astype("float32")

t0 = time.perf_counter()
dists = np.linalg.norm(data - query, axis=1)
true_nearest = set(np.argsort(dists)[:10])
print("brute-force:", (time.perf_counter() - t0) * 1000, "ms")

index = hnswlib.Index(space="l2", dim=64)
index.init_index(max_elements=50000, ef_construction=100, M=16, random_seed=0)
index.add_items(data, np.arange(50000), num_threads=1)  # single-threaded: makes graph construction deterministic

for ef in [10, 50, 100, 300]:
    index.set_ef(ef)
    t0 = time.perf_counter()
    labels, _ = index.knn_query(query, k=10)
    t = (time.perf_counter() - t0) * 1000
    recall = len(set(labels[0]) & true_nearest) / 10
    print(f"ef={ef}: {t:.3f} ms, recall@10={recall:.2f}")
```

```text
brute-force: 16.992 ms

ef=10:  0.054 ms, recall@10=0.10
ef=50:  0.077 ms, recall@10=0.70
ef=100: 0.116 ms, recall@10=1.00
ef=300: 0.349 ms, recall@10=1.00
```

At its fastest setting, the ANN index is over 300x faster than brute-force search, but recovers only 1 of the true 10 nearest neighbors. Raising `ef` — the search-time parameter controlling how much of the index gets explored — trades that speed back for accuracy: by `ef=100`, recall reaches a perfect 1.00, still over 100x faster than brute-force. Past that point, `ef=300` costs three times the search time for no additional recall at all here. This is the tunable trade-off [Retrieval](../rag/retrieval.md#limitations) already named — not a fixed cost, but a dial with a real, verifiable curve behind it.

**A real caveat, worth stating plainly: HNSW's graph construction is itself randomized**, independent of any `numpy` seed. Without pinning `random_seed` and forcing single-threaded construction (`num_threads=1`) as done above, rerunning this exact script produces a *different* graph each time, and recall at a given `ef` genuinely varies run to run — in unseeded repeats of this same example, `ef=100` landed anywhere from 0.70 to 1.00, not reliably the perfect score shown here. The trade-off's *shape* (recall rises with `ef`, cost rises too) is real and stable; the *exact numbers* at a given `ef` are not, unless construction is explicitly pinned the way this example now does.

## Where is it used?

Every production RAG system's retrieval step at real scale, semantic search products, recommendation systems storing user or item embeddings, and any application needing fast similarity search over more vectors than brute-force comparison can handle.

## Advantages

- **Makes similarity search fast at a scale brute-force comparison can't reach**, as the example shows directly — over 100x faster at perfect recall, and far more at lower settings.
- **Combines vector search with familiar database features** — metadata storage, filtering, persistence — rather than requiring a hand-built indexing layer alongside a separate metadata store.
- **The speed/recall trade-off is tunable**, not fixed, letting a system choose where to sit on that curve based on its actual latency and accuracy needs.

## Limitations

- **Approximate search can miss the true best match**, by design — the speed gain comes from accepting a tunable error rate, not a free lunch, exactly as this chapter's `ef=10` result shows starkly (recall of only 0.10).
- **Index build time and memory cost scale with the number of vectors**, and rebuilding a large index after significant data changes isn't instantaneous.
- **Choosing the right index parameters requires knowing the actual recall a use case needs** — there's no universally correct setting, only a trade-off to tune against real requirements.

## Production considerations

- **The `ef` (or equivalent) parameter needs to be tuned against real queries and a real recall target**, not left at a default that may be too aggressive or too conservative for the actual use case.
- **Metadata filtering combined with vector search adds real complexity** — filtering before, during, or after the similarity search can produce meaningfully different results and performance characteristics.
- **Index rebuilds and updates need an operational plan**, the same reproducibility and versioning discipline already established in [ML Workflow](../machine-learning/ml-workflow.md) — a stale index silently serves outdated retrieval results.

## Common mistakes

- **Assuming approximate search is "close enough" without checking recall against real queries**, when the actual miss rate at a given setting — as this chapter's `ef=10` result shows — can be far worse than assumed.
- **Tuning purely for latency without measuring the resulting drop in recall**, the same warning [Retrieval](../rag/retrieval.md#production-considerations) already raised.
- **Treating a vector database as a drop-in replacement for a relational database**, when metadata filtering and consistency guarantees often work differently than in a traditional database.

## Interview questions

### Basic

- What does a vector database do that a traditional database doesn't?
- Why does approximate nearest neighbor search trade off speed against accuracy?

### Intermediate

- In this chapter's example, why did recall improve as `ef` increased, and why did it stop improving past a certain point?
- When would a general-purpose database's vector extension be preferable to a dedicated vector database?

### Advanced

- How would you determine the right recall target for a production RAG system, and how would that target inform your index parameter choices?
- A vector database's index needs to be rebuilt after a large batch of new documents is added. What operational considerations would you plan for during that rebuild?
