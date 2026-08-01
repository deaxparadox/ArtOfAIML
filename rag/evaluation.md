# Evaluation

See [What is RAG](what-is-rag.md) for the two-halves framing (retrieval, generation) this chapter measures separately.

## What is it?

Evaluation, in the RAG context, is the practice of measuring how well a pipeline actually performs — whether retrieval found the right chunks, and separately, whether generation produced a good answer from what it was given.

If a student gets a question wrong, you need to know whether they were handed the wrong page of the book (a retrieval problem) or handed the right page but still answered it incorrectly (a generation problem). Evaluation is the process of checking both, not just the final grade.

## Why does it exist?

[What is RAG](what-is-rag.md#common-mistakes) already named this directly as a common mistake: not measuring retrieval accuracy separately from final answer quality makes debugging much harder, since a good final answer can happen despite bad retrieval, and a bad answer can happen despite good retrieval. Evaluation exists to give a concrete, decomposed answer to "why is this system producing a bad answer," instead of a single pass/fail verdict on the whole pipeline that doesn't say which half needs fixing.

**Retrieval metrics, generation metrics, and end-to-end metrics answer different questions, and none substitutes for the others.** Retrieval metrics measure whether the system found the right information at all, independent of what the model did with it. Generation metrics measure whether the model used good context well, independent of whether retrieval actually supplied it. End-to-end metrics measure what the user actually experiences, but conflate both halves together, and alone don't say which one to fix when something goes wrong.

## How does it work?

- **Retrieval evaluation** typically uses metrics like recall@k (was a relevant document actually in the top-k results) or mean reciprocal rank (how high did the first relevant document rank), computed against a labeled set of query → known-relevant-document pairs.
- **Generation evaluation** checks whether the model's answer is faithful to the retrieved context (doesn't invent claims beyond what was actually given — hallucination in the RAG-specific sense) and relevant to the actual question asked.
- A common, practical technique for generation evaluation at scale is using **another LLM as a judge**, scoring an answer against the retrieved context and the question, since human evaluation of every single response doesn't scale.

## Example

Recall@k, computed directly against a small labeled test set, reusing the corpus from [What is RAG](what-is-rag.md):

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

documents = [
    "The company's return policy allows returns within 30 days of purchase with a valid receipt.",
    "Our support team is available Monday through Friday, 9am to 6pm Eastern Time.",
    "Shipping typically takes 3 to 5 business days within the continental United States.",
    "The warranty covers manufacturing defects for up to one year from the date of purchase.",
]

test_set = [
    ("How long do I have to return something I bought?", 0),
    ("When can I reach customer service?", 1),
    ("How fast will my order arrive?", 2),
    ("Am I covered if the product breaks after 6 months?", 3),
    ("Do you ship internationally?", 2),
]

vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)

def top_k(query, k):
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, doc_vectors)[0]
    return list(np.argsort(-sims)[:k])

recall_at_1 = sum(idx in top_k(q, 1) for q, idx in test_set) / len(test_set)
recall_at_2 = sum(idx in top_k(q, 2) for q, idx in test_set) / len(test_set)
print(recall_at_1, recall_at_2)
```

```text
recall@1: 0.4
recall@2: 0.6
```

Only 2 of 5 test queries retrieved the correct document as the single top result; 3 of 5 had it somewhere in the top 2. "When can I reach customer service?" retrieved the return-policy document first instead of the support-hours one — a genuine retrieval miss caught directly by this metric, not a contrived failure. Notice, too, that building this test set required a judgment call of its own: for "Do you ship internationally?", no document actually answers the question — the shipping document was labeled "correct" as the closest available topic, not a perfect answer. Constructing a fair labeled test set is itself real evaluation work, not a solved prerequisite step.

## Where is it used?

Any RAG system beyond an initial prototype — comparing chunking strategies, retrieval methods, or reranking approaches against each other requires a consistent way to measure whether a change actually helped, rather than judging by a handful of manually inspected examples.

## Advantages

- **Turns "does this seem better" into a specific, comparable number**, the same way a proper evaluation metric does for any ML system, per [ML Workflow](../machine-learning/ml-workflow.md).
- **Separating retrieval and generation metrics tells you which half of the pipeline to actually fix**, instead of one undifferentiated verdict.
- **LLM-as-judge scoring makes evaluating generation quality at scale practical**, where full human review of every response wouldn't be.

## Limitations

- **Building a fair labeled test set is real, nontrivial work** — as the example shows, deciding what counts as "correct" isn't always obvious, especially for queries no single document answers well.
- **Retrieval metrics like recall@k only tell you whether a relevant chunk was retrieved**, not whether the model actually used it correctly once it was.
- **LLM-as-judge evaluation inherits its own judge model's blind spots and biases** — it's a practical, scalable proxy for human judgment, not a perfectly neutral ground truth.

## Production considerations

- **A test set needs to be kept current** as the underlying document collection changes — a query's "correct" answer can become wrong or outdated the same way any other piece of source data can.
- **Evaluation should run on every meaningful pipeline change** — a new chunking strategy, a different embedding model, an added reranking step — the same discipline [ML Workflow](../machine-learning/ml-workflow.md) already established for evaluating any model change.
- **End-to-end metrics alone can mask which component regressed** after a change. A drop in overall answer quality could come from retrieval, generation, or both, and only decomposed metrics can tell which.

## Common mistakes

- **Judging RAG quality from a handful of manually inspected examples** instead of a consistent, repeatable metric computed over a real test set.
- **Measuring only end-to-end answer quality**, without separately checking whether retrieval or generation is the actual source of a problem.
- **Treating an LLM judge's score as objective ground truth**, rather than as a scalable but imperfect proxy for human judgment.

## Interview questions

### Basic

- Why isn't a single end-to-end pass/fail metric enough to evaluate a RAG system?
- What does recall@k measure?

### Intermediate

- Why can a RAG system produce a good final answer even when retrieval performed badly?
- What is LLM-as-judge evaluation, and why is it used instead of human review for every response?

### Advanced

- Constructing a labeled test set for retrieval evaluation involves deciding what counts as a "correct" document for a given query. What makes that harder than it sounds, and how would you handle a query no document answers well?
- A RAG system's end-to-end answer quality dropped after a change. How would you determine whether retrieval or generation is responsible?
