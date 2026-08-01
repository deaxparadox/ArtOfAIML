# Similarity

See [Embeddings](embeddings.md) for the vectors this chapter measures the closeness of.

## What is it?

Similarity is a numerical measure of how close two vectors — typically two embeddings — are to each other: a single number computed directly from the vectors, standing in for "how alike are these two things."

Continuing directly from [Embeddings](embeddings.md)'s "coordinates in meaning space" idea: if embeddings are coordinates, similarity is the specific measurement taken between two points. There's more than one way to measure "closeness," and which one gets picked changes the answer — sometimes dramatically.

## Why does it exist?

[Embeddings](embeddings.md) placed similar things close together in a vector space, but "close together" needs a concrete, computable definition to actually be useful. Similarity exists to supply that definition: a formula that takes two vectors and returns one number, turning "find the most similar item" from a vague spatial intuition into an actual, executable operation — compute the score against every candidate, return the highest.

**Which metric to use is a real decision, not a default to skip past:**

- **Cosine similarity** measures the angle between two vectors, ignoring their magnitude — the standard default for text embeddings, because what usually matters for meaning is direction, not how long the vector happens to be (which can vary with unrelated factors, like sentence length).
- **Euclidean distance** measures the straight-line distance between two points and does care about magnitude — the right choice when actual scale differences carry real information, such as comparing raw numeric feature vectors rather than normalized embeddings.
- **Dot product** is a cheaper, unnormalized version of cosine similarity — mathematically identical to it once vectors are normalized to unit length. Many production vector search systems pre-normalize their stored vectors specifically so they can use the cheaper dot product instead of computing cosine similarity fresh on every query.

## How does it work?

**Cosine similarity**: `dot(A, B) / (||A|| × ||B||)`, ranging from -1 (opposite direction) through 0 (unrelated) to 1 (identical direction).

**Euclidean distance**: `sqrt(sum((aᵢ - bᵢ)²))` — smaller means closer, the opposite direction from similarity, where a *bigger* number means closer. Mixing the two up — treating a smaller similarity score as "closer," the way a smaller distance is — is a genuine, easy source of a bug in ranking logic.

## Example

A case where the two metrics disagree outright:

```python
import numpy as np

A = np.array([1, 1])
B = np.array([3, 3])      # same direction as A, larger magnitude
C = np.array([1, -0.9])   # different direction from A, similar magnitude

def cosine_sim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def euclidean(x, y):
    return np.linalg.norm(x - y)

print(cosine_sim(A, B), euclidean(A, B))  # -> 1.0, 2.83
print(cosine_sim(A, C), euclidean(A, C))  # -> 0.053, 1.9
```

By cosine similarity, `B` is essentially identical to `A` — a score of 1.0, exactly the same direction — while `C` is nearly unrelated (0.053). By Euclidean distance, the ranking flips entirely: `C` is actually closer to `A` (1.9) than `B` is (2.83). Same two candidate vectors, same reference point, opposite conclusion about which one is "more similar" — depending entirely on which metric was chosen.

## Where is it used?

Semantic search (ranking documents by similarity to a query embedding), recommendation systems (finding items similar to what a user already liked), and the retrieval step of retrieval-augmented generation, covered later in this handbook — all of them are, mechanically, "compute a similarity score against many candidates and keep the highest."

## Advantages

- **Turns "how alike are these two things" into a single, comparable, sortable number.**
- **Cosine similarity removes magnitude as a confounding factor**, which matters because embedding magnitude is often an artifact of the model or input length, not a meaningful signal.
- **Cheap to compute** — a similarity score between two vectors is a small, fast calculation, which is exactly what makes comparing against millions of candidates in a search system feasible at all.

## Limitations

- **A high score doesn't guarantee two things are similar in the way that matters for the task** — it only reflects what the embedding model itself learned to place close together, already flagged as a limitation in [Embeddings](embeddings.md).
- **Choosing the wrong metric for the embedding it's applied to produces a silent error.** A metric that assumes normalized vectors, applied to vectors that aren't, can flip a ranking exactly the way the example above shows, without any warning.
- **Similarity alone doesn't scale to searching millions of vectors efficiently.** Computing it against every candidate one at a time is fine for a handful of items, and becomes the exact bottleneck the handbook's later vector databases section exists to solve.

## Production considerations

- **The similarity metric used at search time must match what the embedding model and index were built around.** Mixing cosine similarity with vectors that were never normalized, or vice versa, produces plausible-looking but wrong rankings — not an obvious error.
- **Computing similarity against every stored vector one at a time doesn't scale.** Production systems index vectors for fast approximate search rather than brute-force comparing a query against every candidate.
- **A similarity threshold ("only return results above 0.8") is itself a real decision.** Too high, and legitimately relevant results get excluded; too low, and irrelevant ones get through — the right threshold depends on the embedding model and data, not a universal constant.

## Common mistakes

- **Confusing similarity and distance directionally** — treating a smaller similarity score as "closer," the way a smaller distance is, when for similarity, larger means closer.
- **Applying cosine-similarity assumptions to a metric or vectors where they don't hold**, silently producing the kind of ranking flip shown directly in this chapter's example.
- **Assuming similarity scores are comparable across different embedding models.** As [Embeddings](embeddings.md) already covered, embeddings from different models don't share a space, so neither do similarity scores computed against them.

## Interview questions

### Basic

- What does a cosine similarity score of 1 mean? What about -1 or 0?
- What's the difference between a similarity score and a distance metric?

### Intermediate

- Why is cosine similarity generally preferred over Euclidean distance for comparing text embeddings?
- Why can dot product be used as a cheaper substitute for cosine similarity in some systems?

### Advanced

- Give a concrete example where cosine similarity and Euclidean distance would rank two candidates differently, and explain why.
- Why doesn't brute-force similarity computation scale to searching millions of vectors, and what does that motivate?
