# Reranking

See [Retrieval](retrieval.md) for the initial search this chapter refines.

## What is it?

Reranking is a second, more precise scoring pass applied to the results retrieval already returned — taking the top-k candidates from an initial search and re-ordering them with a slower, more accurate relevance signal, before handing the final selection to the model.

If retrieval is the librarian quickly pulling every book that might be relevant off the shelf using a fast index, reranking is a second, slower pass where someone actually skims each pulled book more carefully to put them in a better order, before handing over just the top few.

## Why does it exist?

[Retrieval](retrieval.md) needs to be fast enough to search over potentially millions of chunks, which is exactly why it relies on cheap methods — approximate nearest neighbor search, compressed or precomputed representations. That speed constraint means retrieval's ranking is a rough cut: good enough to narrow millions of candidates down to a manageable handful, but not necessarily precise enough to guarantee the truly best chunk lands at position 1 rather than position 8. Reranking exists to fix this: since the candidate set is now small — dozens, not millions — it's affordable to apply a slower, more accurate scoring method to just those candidates, one that wouldn't scale to the full corpus but scales fine to a shortlist.

**Whether reranking is worth adding is a real decision.** It adds a real, extra step — another scoring pass, more latency — to every single query, so it's worth adding specifically when initial retrieval's precision is a real bottleneck on final answer quality, not as a reflexive addition to every RAG pipeline. For a corpus where retrieval is usually already accurate in the top few results, reranking adds cost without changing the outcome; simply including a few more retrieved candidates directly in the prompt and letting the model itself sort out relevance can sometimes be a cheaper, adequate alternative, at the cost of more tokens per query.

## How does it work?

Rerankers are typically built as **cross-encoders**: unlike an embedding model, which encodes the query and each document independently so their embeddings can be compared cheaply after the fact, a cross-encoder takes the query and one candidate document together, as a single joint input, and directly outputs a relevance score for that specific pair. This joint processing captures interactions between the query and document that two independently-computed embeddings structurally can't — at the cost of not being reusable or precomputable the way embeddings are, since a fresh score has to be computed for every query-document pair at request time. That cost is exactly why reranking only runs on retrieval's narrowed-down shortlist, never the entire corpus.

## Example

A full cross-encoder isn't practical to demonstrate here, but the same underlying idea — a cheap, compressed initial pass followed by a slower, more information-preserving second pass — is directly verifiable using tools already covered: comparing the compressed (SVD) similarity scores from initial retrieval against the full, uncompressed TF-IDF similarity for the same shortlist.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Our premium plan includes priority customer support and faster response times.",
    "The basic plan covers standard features with email support only.",
    "Enterprise customers get a dedicated account manager and premium onboarding support.",
    "You can upgrade your plan at any time from the billing settings page.",
]
query = "Which plan gives me faster support response times?"

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(documents)
qv = tfidf.transform([query])

svd = TruncatedSVD(n_components=2, random_state=0)
X_reduced = svd.fit_transform(X)
qv_reduced = svd.transform(qv)
initial_scores = cosine_similarity(qv_reduced, X_reduced)[0]  # cheap, compressed
full_scores = cosine_similarity(qv, X)[0]                     # slower, full precision

print("initial retrieval (compressed):")
for doc, score in sorted(zip(documents, initial_scores), key=lambda p: -p[1]):
    print(round(score, 3), doc)

print("\nreranked (full precision):")
for doc, score in sorted(zip(documents, full_scores), key=lambda p: -p[1]):
    print(round(score, 3), doc)
```

```text
initial retrieval (compressed):
0.984 - Our premium plan includes priority customer support and faster response times.
0.892 - Enterprise customers get a dedicated account manager and premium onboarding support.
0.589 - The basic plan covers standard features with email support only.
0.234 - You can upgrade your plan at any time from the billing settings page.

reranked (full precision):
0.649 - Our premium plan includes priority customer support and faster response times.
0.144 - The basic plan covers standard features with email support only.
0.071 - Enterprise customers get a dedicated account manager and premium onboarding support.
0.060 - You can upgrade your plan at any time from the billing settings page.
```

Both passes agree on the top result. They disagree past that: the compressed initial pass ranks the enterprise-support document second; the full-precision pass ranks it third, behind the basic-plan document instead. This isn't a claim that the second pass matches human judgment perfectly — it's a concrete, verified case where a cheap, compressed signal and a more information-preserving one genuinely disagree on ordering beyond the top result, exactly the situation reranking exists to catch before it reaches the model.

## Where is it used?

Any RAG or search pipeline where initial retrieval's top-k accuracy is good but not good enough, and the extra latency of a second scoring pass over a small shortlist is worth the improvement — high-stakes search (legal, medical, code search) more often than casual, low-stakes lookups.

## Advantages

- **Corrects ordering mistakes** that a cheap, scalable initial retrieval pass makes, using a more precise, non-scalable signal.
- **Only runs over a small shortlist, not the whole corpus**, keeping its extra cost bounded regardless of overall corpus size.
- **Captures query-document interactions** that independently-computed embeddings structurally can't.

## Limitations

- **Adds real latency to every query**, since it's an extra scoring step run after initial retrieval, before the model ever sees the result.
- **Can't recover what retrieval missed entirely.** If the truly relevant chunk wasn't in the initial shortlist at all, reranking can only reorder what's there — it can't find what isn't.
- **A more precise signal isn't automatically the objectively "correct" one**, as the example above shows — it's a different, more information-preserving measure, not a guarantee of matching real human relevance judgment.

## Production considerations

- **Reranking cost scales with shortlist size, not corpus size.** Keeping the initial retrieval's top-k small enough to rerank cheaply, while still large enough to likely contain the truly relevant chunk, is a real, tunable trade-off.
- **Reranking adds a sequential step to the request path.** Its latency stacks directly on top of retrieval and generation latency, rather than running in parallel with them.
- **Whether reranking is worth its added latency should be measured against actual answer quality, not assumed.** A corpus where initial retrieval is already reliably accurate in the top few results may not benefit enough to justify the extra step.

## Common mistakes

- **Adding a reranking step to every RAG pipeline by default**, without first checking whether initial retrieval's top-k accuracy actually has room to improve.
- **Assuming reranking can recover a relevant chunk that retrieval failed to include** in its initial shortlist at all.
- **Treating a reranker's score as ground truth relevance**, rather than as a more precise, but still imperfect, signal.

## Interview questions

### Basic

- What does reranking do that initial retrieval doesn't?
- Why can't reranking simply replace retrieval and score the entire corpus directly?

### Intermediate

- What's the structural difference between how an embedding model and a cross-encoder score a query-document pair?
- Why does reranking's cost depend on shortlist size rather than total corpus size?

### Advanced

- A RAG system's answers are still wrong even after adding reranking. What would that suggest about where the actual problem is?
- How would you decide whether the added latency of reranking is actually worth it for a given RAG system?
