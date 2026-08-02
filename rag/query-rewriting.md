# Query Rewriting / Query Expansion

See [Retrieval](retrieval.md) for the search step this chapter improves before it ever runs.

## What is it?

Query rewriting is the practice of transforming a user's original query into a different, often better-phrased query before it's actually used for retrieval — expanding it with related terms, breaking it into sub-questions, or rephrasing it to better match how relevant documents are likely worded.

[Retrieval](retrieval.md) showed retrieval finding the right document even when the query was phrased differently from the document's own wording. Query rewriting is a deliberate, upfront step to make that kind of mismatch less likely to matter in the first place — rephrasing or expanding the query itself before it's even sent to the retriever, rather than relying entirely on retrieval to bridge the gap.

## Why does it exist?

[Chunking](chunking.md) and [Retrieval](retrieval.md) both already showed real, honest failure modes tied to a mismatch between how a query is phrased and how relevant content is phrased or structured. Query rewriting exists to attack that mismatch from the other direction: instead of only improving how documents are indexed or chunked, or how retrieval scores a match, it improves the query itself before retrieval ever runs — turning a short, ambiguous, or narrowly-worded query into one, or several, that are more likely to actually match relevant content.

**Query expansion vs. query decomposition — two different, real techniques.** Query expansion adds related terms to a single query, helping even keyword-based retrieval catch a paraphrase it would otherwise miss entirely. Query decomposition breaks a single complex question into several simpler sub-questions, each retrieved separately and then combined — useful when one query is actually asking several distinct things that no single retrieved chunk could answer alone.

## How does it work?

**Query expansion** adds synonyms or related terms to the original query — via a thesaurus, or by asking an LLM to suggest related phrasings — broadening what the retrieval step can match against. **Query decomposition** uses an LLM to break a compound question into its distinct sub-questions, retrieving separately for each before combining the results. Both add an extra step, and often an extra model call, *before* retrieval happens — trading added latency and cost for improved retrieval accuracy, the mirror image of [Reranking](reranking.md)'s extra step added *after* retrieval instead.

## Example

The same documents from [What is RAG](what-is-rag.md), with a query phrased in a way that shares no vocabulary with any of them at all:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "The company's return policy allows returns within 30 days of purchase with a valid receipt.",
    "Our support team is available Monday through Friday, 9am to 6pm Eastern Time.",
    "Shipping typically takes 3 to 5 business days within the continental United States.",
    "The warranty covers manufacturing defects for up to one year from the date of purchase.",
]

vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)

def rank(query):
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, doc_vectors)[0]
    return sorted(zip(documents, sims), key=lambda p: -p[1])

original_query = "Can I send this back?"
rewritten_query = "Can I send this back? What is your return and refund policy?"

for label, query in [("original", original_query), ("rewritten", rewritten_query)]:
    print(f"{label} query {query!r}:")
    for doc, score in rank(query):
        print(round(score, 3), doc)
```

```text
original query "Can I send this back?":
0.0 - (every document, all tied at zero)

rewritten query "Can I send this back? What is your return and refund policy?":
0.335 - The company's return policy allows returns within 30 days of purchase with a valid receipt.
0.164 - Our support team is available Monday through Friday, 9am to 6pm Eastern Time.
0.0   - Shipping typically takes 3 to 5 business days within the continental United States.
0.0   - The warranty covers manufacturing defects for up to one year from the date of purchase.
```

The original query fails completely — it shares no vocabulary with any document, so every similarity score is exactly zero. This isn't a subtle ranking problem; retrieval had nothing to work with at all. Adding "return and refund policy" to the query — the kind of expansion an LLM could generate automatically from the original short question — gives retrieval real terms to match against, correctly surfacing the return-policy document with a clear score lead over everything else.

## Where is it used?

Any RAG system fielding short, casual, or ambiguous user queries — chat-style interfaces especially, where users rarely phrase questions the way source documents are written — and any system where a single user question genuinely spans multiple distinct sub-topics.

## Advantages

- **Fixes retrieval failures at the source**, as the example shows directly — a query that matched nothing at all became retrievable with no change to the underlying documents or index.
- **Complements rather than replaces other retrieval improvements** — expansion helps keyword-based retrieval specifically, while [Reranking](reranking.md) and better [Chunking](chunking.md) address different parts of the same overall pipeline.
- **Decomposition handles genuinely compound questions** that a single retrieval pass, however well-tuned, couldn't answer from one matched chunk.

## Limitations

- **Adds latency and cost before retrieval even starts** — an LLM call to rewrite or decompose the query, on top of everything that follows it.
- **A bad rewrite can make retrieval worse, not better** — expanding a query with the wrong related terms can drift it away from what the user actually meant.
- **Doesn't fix a badly chunked or poorly indexed corpus.** If the relevant content was never captured well in the first place, no amount of query rewriting recovers it.

## Production considerations

- **Rewriting quality needs to be evaluated the same way retrieval itself is**, per [Evaluation](evaluation.md) — a rewriting step that quietly makes results worse is a real regression, not a neutral addition.
- **The added latency compounds with every other step in the pipeline** — [Reranking](reranking.md)'s extra pass after retrieval, and now a rewriting pass before it, both stack onto end-to-end response time.
- **Query decomposition's sub-question results need a clear combination strategy** — simply concatenating multiple retrieved sets without deduplication or synthesis can flood the final prompt with redundant context.

## Common mistakes

- **Assuming a query "should" retrieve correctly without rewriting**, and blaming the retrieval algorithm when the actual problem, as this chapter's example shows, is that the query and the documents simply share no matchable terms.
- **Rewriting every query by default**, even simple ones that already retrieve well, adding cost and latency with no corresponding benefit.
- **Not evaluating rewritten queries separately from the original ones**, missing cases where the rewrite actually hurt retrieval instead of helping it.

## Interview questions

### Basic

- What's the difference between query expansion and query decomposition?
- Why did the original query in this chapter's example fail to retrieve anything at all?

### Intermediate

- Why does query rewriting add latency, and what does that cost buy in return?
- When would query decomposition be necessary, rather than expansion alone?

### Advanced

- How would you detect that a query-rewriting step is making retrieval worse for a subset of real user queries, rather than better?
- Design an evaluation approach that separately measures the quality of query rewrites from the quality of the retrieval that follows them.
