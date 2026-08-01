# TF-IDF / Bag-of-Words

See [Tokenization](tokenization.md) for the tokens this representation counts, and [Embeddings](../embeddings/embeddings.md) for the dense-vector alternative this chapter contrasts with.

## What is it?

Bag-of-words represents text as a vector of word counts, ignoring grammar and word order entirely — just which words appeared, and how many times. TF-IDF (Term Frequency–Inverse Document Frequency) refines this by weighting each word by how distinctive it is to a specific document, not just how often it appears.

If tokenization breaks text into pieces and embeddings place those pieces in a meaning space, bag-of-words and TF-IDF are a much simpler, older way of turning text into numbers: a sparse vector counting which words showed up, weighted so that common, unhelpful words count for less than rare, distinctive ones.

## Why does it exist?

Before embeddings existed, there needed to be some way to turn text into numbers a model or similarity computation could work with, and bag-of-words was one of the earliest, simplest answers: literally count word occurrences. Raw counts have an obvious problem — extremely common words ("the," "is," "and") show up in nearly every document and dominate raw counts, drowning out the words that actually distinguish one document from another. TF-IDF exists to fix exactly that: it down-weights words that appear in most documents (low "inverse document frequency" — not distinctive) and up-weights words that appear often in one document but rarely elsewhere.

**TF-IDF vs. dense embeddings, as a real decision — already seen in practice.** [Retrieval](../rag/retrieval.md) already showed this trade-off directly: raw TF-IDF found an exact error-code match unambiguously, while a compressed semantic embedding nearly missed it entirely. TF-IDF is the right choice when exact term matching matters — a specific code, name, or rare term — is fast and simple to compute with no training required, and stays fully interpretable, since every dimension corresponds to an actual word. Dense embeddings are the right choice when semantic similarity matters more than exact wording, at the cost of losing that exact-term specificity and interpretability.

## How does it work?

**Term Frequency (TF)** is how often a word appears in a specific document. **Inverse Document Frequency (IDF)** measures how rare a word is across the entire document collection — a word appearing in almost every document gets a low IDF; a word appearing in only a few gets a high one. The final **TF-IDF score** is TF times IDF: high specifically for a word that appears often in one document but rarely across the collection. The result is a sparse vector — one dimension per unique word in the vocabulary, mostly zeros, since any single document only uses a small fraction of it.

## Example

The actual IDF weighting, verified directly on three short documents:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "the cat sat on the mat",
    "the dog sat on the rug",
    "the error code reported was unusual",
]

tfidf_vec = TfidfVectorizer()
tfidf_vec.fit(documents)
idf = dict(zip(tfidf_vec.get_feature_names_out(), tfidf_vec.idf_))
for word in ["the", "sat", "cat", "error", "unusual"]:
    print(word, idf[word])
```

```text
the:     1.000  (appears in all 3 documents)
sat:     1.288  (appears in 2 of 3 documents)
cat:     1.693  (appears in 1 of 3 documents)
error:   1.693  (appears in 1 of 3 documents)
unusual: 1.693  (appears in 1 of 3 documents)
```

The word appearing in every document gets the lowest possible IDF; words appearing in only one document all get the same, higher IDF. Comparing the final weighted vector for the first document makes the effect concrete: `"cat"` and `"mat"` (each appearing in only 1 of 3 documents) score 0.469, higher than `"on"` and `"sat"` (appearing in 2 of 3 documents) at 0.356 — identical raw counts, different final weight, purely from rarity. `"the"` itself still ends up with the highest weight in this specific document (0.554) — not because it's undervalued, but because it appears twice in this one document, and TF-IDF discounts common words, it doesn't erase them. That's the honest nuance: TF-IDF balances frequency against distinctiveness, it doesn't guarantee common words always rank last.

## Where is it used?

Classical search and retrieval (exactly what [Retrieval](../rag/retrieval.md) demonstrated), keyword-based document ranking, and any baseline text representation used for comparison against a newer embedding-based approach, since it needs no training and is fully interpretable.

## Advantages

- **No training required** — computed directly and deterministically from the document collection.
- **Fully interpretable** — every dimension corresponds to an actual word, unlike a dense embedding's opaque coordinates.
- **Handles exact, rare terms reliably**, exactly where dense embeddings can under-represent them, as [Retrieval](../rag/retrieval.md) already showed.

## Limitations

- **Ignores word order and grammar entirely.** "The cat chased the dog" and "the dog chased the cat" produce the exact same bag-of-words vector.
- **Vocabulary size grows with the document collection**, and the resulting sparse vectors can become very large for a big corpus, unlike a fixed-size dense embedding.
- **Doesn't capture meaning or synonymy at all.** "Car" and "automobile" are entirely unrelated dimensions to TF-IDF, no matter how semantically similar they are.

## Production considerations

- **The vocabulary is fixed at fit time.** A word never seen in the training corpus has no dimension at all in a new document's vector — unlike subword tokenization's [Tokenization](tokenization.md) fallback behavior, TF-IDF simply can't represent it.
- **Recomputing IDF values as the document collection grows or changes** is a real maintenance cost — stale IDF weights, from a collection that's since changed, produce a subtly wrong notion of what counts as "rare."
- **Sparse vector storage and computation scales with vocabulary size**, which matters directly for a large, growing corpus the way dense embeddings' fixed dimensionality doesn't.

## Common mistakes

- **Assuming TF-IDF captures meaning the way embeddings do.** It only captures term frequency and rarity — "great" and "excellent" share no similarity in a TF-IDF vector.
- **Expecting TF-IDF to always rank common words last**, when a word repeated many times within one document can still outweigh a rarer word's IDF advantage, as this chapter's example shows directly.
- **Using a stale vocabulary or stale IDF values** against a document collection that's meaningfully changed since they were computed.

## Interview questions

### Basic

- What's the difference between raw word counts and TF-IDF?
- What does a high IDF value indicate about a word?

### Intermediate

- Why can a common word still end up with the highest weight in a specific document's TF-IDF vector?
- Why does TF-IDF fail on a word that never appeared in the original document collection?

### Advanced

- Why did TF-IDF outperform a compressed semantic embedding on the error-code query in the Retrieval chapter, and what does that imply about when to combine both approaches?
- Design a hybrid retrieval system using both TF-IDF and dense embeddings. What would each contribute, and how would you combine their scores?
