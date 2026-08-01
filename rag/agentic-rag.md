# Agentic RAG

See [Agent Architecture](../agents/agent-architecture.md) for the observe/decide/act loop this chapter applies to retrieval, and [What is RAG](what-is-rag.md) for the fixed pipeline it replaces.

## What is it?

Agentic RAG is a RAG system built as an agent loop rather than a fixed pipeline — instead of always retrieving once and generating once, the model can decide to retrieve again, rewrite its own query, critique its own draft answer, or stop and answer once it's confident, repeating the observe/decide/act loop from [Agent Architecture](../agents/agent-architecture.md) with retrieval as one of the available actions.

The standard pipeline from [What is RAG](what-is-rag.md) is a fixed sequence: chunk, embed, retrieve, optionally rerank, generate, done. Agentic RAG replaces that fixed sequence with the same loop already covered in [Agent Architecture](../agents/agent-architecture.md), where "retrieve" becomes one tool among several the model can choose to call, more than once, based on what it's learned so far.

## Why does it exist?

[What is RAG](what-is-rag.md#limitations) already named the problem this exists to solve: "only as good as the retrieval step — if the wrong document is retrieved, the model is working from the wrong context, no matter how good the model itself is." A fixed pipeline retrieves exactly once and has no way to recover if that one attempt was wrong or incomplete. Agentic RAG exists to give the system a way to notice that and act on it: if the retrieved context doesn't actually answer the question, the model can rewrite the query — exactly the technique from [Query Rewriting](query-rewriting.md) — and retrieve again, rather than being stuck with whatever the single retrieval pass happened to return.

**When a fixed pipeline is enough, versus when agentic RAG earns its complexity.** A fixed pipeline is simpler, cheaper, and faster — one retrieval call, one generation call, predictable latency and cost. It's the right choice when queries are usually simple and well-served by a single retrieval pass, which is most of the time for a narrow, well-indexed knowledge base. Agentic RAG earns its added complexity — more model calls, unpredictable latency, the same runaway-loop and cost risks already covered in [Agent Architecture](../agents/agent-architecture.md#production-considerations) — specifically when queries are frequently complex, ambiguous, or poorly served by a single attempt, and a wrong answer is costly enough to justify the system trying harder before giving up.

## How does it work?

The agent observes the current state — the question, and any retrieval attempts made so far — and decides an action: retrieve with the current or a rewritten query, critique its own draft answer against the retrieved context, or finish and return an answer. If the retrieved context is judged insufficient, it rewrites the query and retrieves again, the same loop-with-a-tool-call pattern from [Agent Architecture](../agents/agent-architecture.md) and [Tool Calling](../agents/tool-calling.md), with retrieval itself as one of the callable tools. This continues, bounded by a step limit, until the agent is confident enough to answer or the limit is reached.

## Example

A real control loop chaining together the exact retrieval results already verified in [What is RAG](what-is-rag.md) and [Query Rewriting](query-rewriting.md) — the scripted decision of *when* to rewrite stands in for an LLM's judgment, the retrieval and scoring are real:

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

def retrieve(query):
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, doc_vectors)[0]
    best_idx = sims.argmax()
    return documents[best_idx], sims[best_idx]

CONFIDENCE_THRESHOLD = 0.1

def agentic_retrieve(query, rewrite, max_attempts=2):
    current_query = query
    for attempt in range(max_attempts):
        doc, score = retrieve(current_query)
        print(current_query, "->", round(score, 3))
        if score >= CONFIDENCE_THRESHOLD:
            return doc, score
        current_query = rewrite(current_query)
    return doc, score

rewrite = lambda q: q + " What is your return and refund policy?"
doc, score = agentic_retrieve("Can I send this back?", rewrite)
```

```text
Can I send this back? -> 0.000
Can I send this back? What is your return and refund policy? -> 0.335
```

The first attempt scores 0.000 — below the confidence threshold — so the loop doesn't stop and hand the model an empty or irrelevant context. It rewrites the query and retries, this time scoring 0.335, well above the threshold, and returns the correctly retrieved return-policy document. A fixed, single-shot pipeline would have generated an answer from the first attempt's empty result; this loop noticed the failure and recovered from it.

## Where is it used?

Customer support assistants handling casually-phrased questions, research assistants answering complex, multi-part questions no single retrieval pass could serve, and any RAG system where retrieval quality varies enough across real queries that a single fixed attempt isn't reliable enough on its own.

## Advantages

- **Recovers from a failed retrieval attempt**, as the example shows directly, instead of generating an answer from empty or irrelevant context.
- **Can decompose a complex question into sub-questions** and retrieve for each separately, handling queries a single retrieval pass structurally couldn't serve.
- **Adapts its effort to the query** — a simple, well-matched query can still resolve in one attempt, while a harder one gets additional attempts as needed.

## Limitations

- **Every additional attempt is an additional retrieval and model call**, multiplying the cost and latency [What is RAG](what-is-rag.md#limitations) already flagged for a single RAG request.
- **Needs a real confidence signal to decide when to retry** — a similarity score threshold, as in the example, is a simple one, but a poorly calibrated threshold can trigger unnecessary retries or stop too early on a genuinely bad result.
- **Inherits every agent risk from [Agent Architecture](../agents/agent-architecture.md)** — a runaway loop, compounding errors across attempts — now specifically applied to retrieval instead of a general tool-calling loop.

## Production considerations

- **A hard step limit is not optional**, the same safeguard [Agent Architecture](../agents/agent-architecture.md#production-considerations) already established — a query that never crosses the confidence threshold needs to fail gracefully, not retry indefinitely.
- **Cost and latency become genuinely unpredictable per query**, unlike a fixed pipeline's stable cost — worth monitoring per-query, not just as an average across all traffic.
- **The confidence threshold itself needs tuning and monitoring**, the same way any anomaly or decision threshold does elsewhere in this handbook — too low, and bad retrievals get accepted anyway; too high, and good ones get needlessly retried.

## Common mistakes

- **Adding agentic retries to every RAG system by default**, when a fixed pipeline already serves most queries well and the added cost buys little.
- **Using a confidence threshold that was never validated against real query outcomes**, rather than checked, the way this chapter's own threshold was set against a verified example.
- **Not bounding the number of retry attempts**, risking the same runaway-loop cost already covered for agents generally.

## Interview questions

### Basic

- What does agentic RAG add to the fixed pipeline from What is RAG?
- What signal does the loop in this chapter's example use to decide whether to retry?

### Intermediate

- Why does a fixed, single-shot RAG pipeline have no way to recover from a bad retrieval attempt?
- Why does agentic RAG's cost and latency become harder to predict than a fixed pipeline's?

### Advanced

- How would you tune and validate a confidence threshold for deciding when to retry retrieval, rather than picking one arbitrarily?
- Design a step-limit and fallback strategy for an agentic RAG system so that a query which never crosses the confidence threshold still fails gracefully.
