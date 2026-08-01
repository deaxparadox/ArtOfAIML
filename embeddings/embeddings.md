# Embeddings

See [Tokenization](../nlp/tokenization.md) for the token IDs this chapter's representation improves on.

## What is it?

An embedding is a vector of numbers that represents a piece of data — a word, a sentence, an image, a product — placed in a continuous space where distance and direction carry meaning: similar things end up close together.

Think of embeddings as coordinates in a "meaning space." The same way a city's coordinates place it relative to every other city, an embedding places a word or sentence relative to every other one, such that being close in that space means being similar in meaning, not just similar in spelling.

## Why does it exist?

[Tokenization](../nlp/tokenization.md) converts text into integer token IDs — but those IDs carry no relationship information on their own. Two semantically similar words could easily land on two arbitrary, unrelated numbers. A model needs a representation where "similar meaning" translates into "similar in the actual numbers," because that's what lets it generalize: something learned about one word can transfer to a similar word the model has seen less often, but only if their representations are actually close together.

One-hot encoding (from [Feature Engineering](../machine-learning/feature-engineering.md)) was an earlier attempt at representing categories numerically, and it breaks down at exactly this point: for a vocabulary of tens of thousands of words, a one-hot vector is enormous and mostly zeros, and every word ends up exactly as "different" from every other word — "cat" is precisely as distant from "dog" as it is from "bicycle." Embeddings exist to fix this directly: a dense, much lower-dimensional vector, learned from data rather than designed by hand, where "cat" and "dog" end up close together because they appear in similar contexts, and "bicycle" ends up farther away.

**When embeddings are worth the added complexity, and when they aren't:** one-hot encoding is still fine for a small number of categories with no real relationship between them — a handful of plan tiers, as in [Feature Engineering](../machine-learning/feature-engineering.md)'s own example. Embeddings earn their complexity specifically when there are many items and real similarity structure worth capturing between them — words, products, users — because that's exactly what one-hot encoding is structurally incapable of representing.

## How does it work?

An embedding model is trained so that items appearing in similar contexts end up with similar vectors — "a word is known by the company it keeps" is the founding intuition behind this (the distributional hypothesis). Modern embedding models (word2vec historically, transformer-based sentence embeddings today) learn this from very large corpora using a neural network. This chapter's example instead uses a simpler, classical technique — TF-IDF followed by dimensionality reduction — to demonstrate the same underlying idea without needing a large pretrained model.

## Example

Building a toy embedding for six short sentences across two unrelated topics (pets, the stock market), using TF-IDF + SVD (a classical technique sometimes called Latent Semantic Analysis):

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

sentences = [
    "The cat sat on the mat.",
    "My cat and dog are best friends.",
    "The dog chased the cat around the yard.",
    "The stock market fell sharply today.",
    "Investors are worried about the stock market.",
    "The market crash affected many investors.",
]

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(sentences)

svd = TruncatedSVD(n_components=2, random_state=0)
embeddings = svd.fit_transform(X)

sims = cosine_similarity(embeddings)
print(sims[0, 1])  # cat sentence vs. another cat/dog sentence
print(sims[0, 3])  # cat sentence vs. a stock-market sentence
```

```text
0.957
0.234
```

Two sentences about pets land at a cosine similarity of 0.957 — close together in the embedding space. A pet sentence and a stock-market sentence land at 0.234 — far apart. Nothing told the model "these are about animals" and "these are about finance" directly; the closeness, or distance, came entirely from which words tend to appear together across the sentences.

## Where is it used?

Semantic search and retrieval — finding documents by meaning rather than exact keyword match, the basis of retrieval-augmented generation covered later in this handbook — recommendation systems (representing users and products in the same space so "similar" items can be found by proximity), and as the direct input to the similarity comparisons covered in the next chapter.

## Advantages

- **Captures real similarity structure** that one-hot encoding cannot represent at all.
- **Dense and low-dimensional** compared to one-hot vectors, which matters directly for storage and search speed at scale.
- **Learned from data, not hand-designed** — relationships like synonyms and related concepts emerge automatically from how items co-occur, without anyone manually specifying them.

## Limitations

- **An embedding model's notion of "similar" is entirely shaped by its training data.** A model trained only on English news text won't produce meaningful embeddings for slang, another language, or a specialized technical vocabulary it never saw.
- **Not directly interpretable.** A one-hot vector's meaning is obvious from its structure; an embedding's individual numbers, on their own, mean nothing to a person.
- **Classical techniques capture less than modern neural ones.** The TF-IDF + SVD example above captures topic-level similarity reasonably well, but it wouldn't recognize that "the market crashed" and "stocks plummeted" mean nearly the same thing despite sharing almost no words — a modern neural embedding model would.

## Production considerations

- **Embeddings must be generated consistently, at indexing time and query time, with the exact same model and version.** Comparing an embedding from one model version against one from another produces numbers that aren't meaningfully comparable at all — and nothing about the comparison errors out to warn you.
- **Storing and searching embeddings at scale is its own infrastructure problem**, covered directly in [Vector Databases](../vector-databases/vector-databases.md).
- **Embedding generation has a real, ongoing cost** — compute for a self-hosted model, or an API cost for a hosted embedding service — that scales with data volume, worth budgeting for explicitly the same way token costs matter for [Tokenization](../nlp/tokenization.md).

## Common mistakes

- **Comparing embeddings produced by two different models (or two versions of the same model) as if they live in the same space.** They don't, and the resulting similarity numbers are meaningless.
- **Assuming a general-purpose embedding model will work well on a specialized domain** — legal text, medical records, source code — without checking whether its domain-specific vocabulary and relationships are actually well represented.
- **Treating embedding similarity as a guarantee of correctness rather than a heuristic.** Two texts can be superficially similar in embedding space while differing in exactly the detail that matters for the actual task.

## Interview questions

### Basic

- What is an embedding, and how is it different from a one-hot encoded vector?
- Why does "distance" between two embeddings matter?

### Intermediate

- Why can't you meaningfully compare embeddings produced by two different models?
- What does it mean for an embedding model to have learned its representations from data, rather than by hand-designed rules?

### Advanced

- Why might an embedding model trained on general web text perform poorly on a specialized domain like legal or medical text?
- The TF-IDF + SVD example in this chapter captures topic-level similarity but misses deeper meaning. What kind of sentence pair would it fail on that a modern neural embedding model would get right?
