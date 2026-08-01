# Retrieval

See [Chunking](chunking.md) for how a document collection becomes the pieces this chapter searches over.

## What is it?

Retrieval is the step in a RAG pipeline that takes a query's embedding and finds the most relevant chunks from an indexed document collection — the actual search operation that decides which chunks get handed to the model.

If chunking decided how big each page of the book is, retrieval is the librarian who, given the exam question, flips through the already-indexed book and pulls out the specific pages most likely to help answer it.

## Why does it exist?

[What is RAG](what-is-rag.md) already showed the retrieval half working directly, on four documents. [Chunking](chunking.md) established that real documents get broken into many small pieces before indexing — a real corpus has potentially millions of chunks, not four. Retrieval exists to solve the actual search problem at that scale: given a query embedding, find the closest chunks among potentially millions, without brute-force comparing against every single one every time, exactly the scaling limit [Similarity](../embeddings/similarity.md) already raised.

**Exact vs. approximate search, as a real decision:** exact search — comparing the query against every stored vector and keeping the top matches — always finds the true best matches, but its cost grows linearly with the number of stored chunks, fine for thousands, a real bottleneck at millions. Approximate nearest neighbor (ANN) search trades a small, tunable chance of missing the true best match for dramatically faster search, and is the standard choice for any large-scale production system — "close enough, fast" beats "exactly right, slow" for the vast majority of RAG use cases.

**Semantic vs. keyword retrieval is a second, separate decision**, and the example below shows exactly why it matters: pure semantic (embedding-based) search can lose specificity on exact terms — error codes, product SKUs, precise names — that a traditional keyword search would catch directly. Hybrid retrieval, combining both, exists specifically to cover each approach's blind spot with the other.

## How does it work?

1. Embed the query using the exact same embedding model used to embed the indexed chunks — mismatched models produce meaningless comparisons, as [Embeddings](../embeddings/embeddings.md) already covered.
2. Compare the query embedding against stored chunk embeddings using a similarity metric ([Similarity](../embeddings/similarity.md)).
3. Return the top-k most similar chunks.
4. At scale, step 2 uses an approximate nearest neighbor index rather than brute-force comparison, to make this fast.

## Example

A query about a specific error code, compared under keyword-based search and under the same semantic (TF-IDF + SVD) approach from [Embeddings](../embeddings/embeddings.md):

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "To reset your password, go to account settings and click forgot password.",
    "If you see error code E-4471, restart the device and reconnect to Wi-Fi.",
    "Our mobile app supports both iOS and Android devices running recent versions.",
    "Battery life varies depending on screen brightness and usage patterns.",
]
query = "What does error E-4471 mean?"

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(documents)
qv = tfidf.transform([query])
keyword_sims = cosine_similarity(qv, X)[0]

svd = TruncatedSVD(n_components=2, random_state=0)
X_reduced = svd.fit_transform(X)
qv_reduced = svd.transform(qv)
semantic_sims = cosine_similarity(qv_reduced, X_reduced)[0]
```

```text
keyword-based:
0.394 - If you see error code E-4471, restart the device and reconnect to Wi-Fi.
0.0   - (every other document)

semantic (SVD-compressed):
0.999 - To reset your password, go to account settings and click forgot password.
0.999 - If you see error code E-4471, restart the device and reconnect to Wi-Fi.
0.031 - Battery life varies depending on screen brightness and usage patterns.
```

Keyword search finds the right document unambiguously — the exact term "E-4471" appears nowhere else, so its score is the only nonzero one. Semantic search, compressed down to 2 dimensions the same way the [Embeddings](../embeddings/embeddings.md) example was, essentially fails here: the error-code document and the unrelated password-reset document score almost identically. (This particular gap is exaggerated by using only 2 dimensions on a tiny corpus — a real semantic model would do better — but the underlying phenomenon is genuine: dense compression can wash out an exact, rare term precisely because it's rare, which is the opposite of a keyword search's strength.)

## Where is it used?

Every RAG system's core query path, and any semantic search feature more generally — internal documentation search, code search, customer support knowledge bases.

## Advantages

- **Scales to a large indexed corpus** via approximate nearest neighbor search, instead of a linear scan over everything.
- **Finds relevant content by meaning**, not just exact wording, when semantic retrieval is used — recovering a relevant chunk even when the query is phrased very differently.
- **Hybrid retrieval covers both approaches' blind spots** — exact terms via keyword search, paraphrased meaning via semantic search.

## Limitations

- **Pure semantic retrieval can lose exact-term specificity**, as the example above shows directly — a real, known weakness of dense embedding-based search on rare, specific tokens.
- **Approximate nearest neighbor search can miss the true best match** by design — the speed gain comes from accepting a small, tunable error rate, not from a free lunch.
- **Retrieval quality depends entirely on how well the corpus was chunked and embedded** — a good retrieval algorithm can't recover a fact that was cut in half at a chunk boundary, or embedded by a mismatched model.

## Production considerations

- **Choosing semantic-only, keyword-only, or hybrid retrieval is a real decision** that should be checked against actual queries the system needs to handle — a knowledge base full of product codes and error IDs needs keyword coverage; a knowledge base of narrative documentation may not.
- **Approximate nearest neighbor index parameters trade recall for speed directly** — tuning them too aggressively for speed can silently drop relevant results below the threshold that would surface them.
- **Retrieval latency adds directly to end-to-end response time**, on top of the model's own generation time — worth measuring and budgeting for separately from model latency.

## Common mistakes

- **Assuming semantic retrieval alone is sufficient** without checking whether the corpus contains exact terms — codes, IDs, names — that a purely dense approach might underweight.
- **Treating retrieval as solved once it "returns something"**, without checking whether the top results are actually the most relevant ones for real queries.
- **Tuning an approximate search index purely for latency**, without checking the resulting drop in recall against real queries first.

## Interview questions

### Basic

- What does the retrieval step in RAG actually do?
- Why doesn't brute-force comparison against every chunk scale to a large document collection?

### Intermediate

- Why might semantic (embedding-based) retrieval fail on a query containing a specific error code or product ID?
- What does approximate nearest neighbor search trade away in exchange for speed?

### Advanced

- Design a retrieval strategy for a knowledge base that contains both narrative documentation and structured error codes. What would you combine, and why?
- How would you measure whether a drop in an ANN index's recall parameter is actually hurting real user queries, versus just being a theoretical risk?
