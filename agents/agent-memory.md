# Agent Memory

See [Context Window](../llms/context-window.md) for the short-term limit this chapter's memory extends beyond, and [Vector Databases](../vector-databases/vector-databases.md) for the storage this chapter's retrieval is built on.

## What is it?

Agent memory is the mechanism by which an agent retains information across turns or across sessions — beyond the immediate conversation history in its current context window — so it can recall a user's stated preference, a past decision, or a learned fact without needing it re-explained every time.

A context window is short-term memory: everything currently visible in the conversation. Agent memory extends that with a separate, persistent store the agent can explicitly write to and read from, the way a person jots something in a notebook to remember it after the current conversation ends.

## Why does it exist?

[Context Window](../llms/context-window.md) established the hard limit on how much fits in a single request, and the trade-offs for managing overflow within *one* conversation. Agent memory exists to solve a related but different problem: what happens *across* conversations, once the current context window is gone entirely — a new session starts, hours or days later? Without a separate memory mechanism, every new conversation starts from zero: no user preferences, no prior decisions, no continuity. That's a genuine limitation for any agent meant to be used repeatedly over time, not just once.

**Three real types of long-term memory, each solving a different problem.** **Episodic** memory records specific past events — "the user asked about the return policy last Tuesday and seemed frustrated." **Semantic** memory holds general facts learned over time, not tied to one event — "this user prefers email over phone contact." **Procedural** memory captures a learned pattern of *how* to do something — "this user always wants a summary before details." Which type an agent needs depends on what it's actually being asked to remember: a simple FAQ bot needs none of this; a long-running personal assistant benefits from all three.

## How does it work?

Memory has to be written explicitly — the agent, or a separate process, decides something is worth remembering and stores it — and read explicitly, retrieved when relevant to a new request. It doesn't happen automatically just from having a conversation. Long-term memory is commonly stored as embeddings in a vector database, exactly as covered in [Vector Databases](../vector-databases/vector-databases.md): retrieving a relevant past memory reuses the same retrieval mechanism [Retrieval](../rag/retrieval.md) already established, including the same keyword-vs-semantic choice — often semantic in production, so a paraphrased query still matches a differently-worded memory. Memory retrieval carries the same real trade-off as any retrieval: too little retrieved context misses relevant history; too much wastes tokens, per [Context Window](../llms/context-window.md), on irrelevant memories.

## Example

A small memory store about one user, with a new conversation turn retrieving the most relevant past memory. This uses plain TF-IDF — keyword-based retrieval, the same honestly-named limitation [Retrieval](../rag/retrieval.md#example) already covered — not the embedding-based semantic search a production memory store would more likely use:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

memories = [
    "User prefers email contact over phone calls.",
    "User asked about the return policy on March 3rd and was frustrated with long wait times.",
    "User's account is on the Pro plan, billed annually.",
    "User mentioned they are based in the UK and orders often take longer to arrive.",
]

vectorizer = TfidfVectorizer()
memory_vectors = vectorizer.fit_transform(memories)

def retrieve_memory(query):
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, memory_vectors)[0]
    return sorted(zip(memories, sims), key=lambda p: -p[1])

for mem, score in retrieve_memory("Can you help me with a return?"):
    print(round(score, 3), mem)
```

```text
0.379 User asked about the return policy on March 3rd and was frustrated with long wait times.
0.0 User prefers email contact over phone calls.
0.0 User's account is on the Pro plan, billed annually.
0.0 User mentioned they are based in the UK and orders often take longer to arrive.
```

The new conversation's query correctly surfaces the one relevant memory — this user already had a frustrating return experience — while the other three memories, genuinely irrelevant to this specific request, score exactly zero. That correct match here comes entirely from the literal shared word "return" between the query and the memory — exactly the keyword-matching mechanism, with the same blind spot to a paraphrase using no shared words that [Retrieval](../rag/retrieval.md#example) already demonstrated directly. An agent with this retrieved memory can respond with real continuity ("I see your last return took longer than expected — let me make sure this one goes faster"), instead of treating this as the very first interaction with this user.

## Where is it used?

Personal assistants and customer support agents that need continuity across sessions, agents that adapt their behavior based on a user's past preferences, and any long-running agent where "start over from zero every conversation" would be a genuinely worse experience than remembering context.

## Advantages

- **Provides continuity across sessions**, as the example shows directly, rather than starting every conversation with no history at all.
- **Separates what's worth remembering long-term from what's only relevant to the current conversation**, keeping the context window focused on the immediate task.
- **Reuses the same retrieval mechanism already established for RAG**, rather than requiring an entirely new technique.

## Limitations

- **Memory has to be curated, not automatic.** Deciding what's actually worth storing long-term — and what would just be noise — is a real, ongoing design decision, not something that happens for free.
- **Stale or incorrect memories can mislead an agent just as badly as a stale RAG index can**, per [Vector Databases](../vector-databases/vector-databases.md#production-considerations) — a memory that was true last year and isn't anymore is actively harmful, not neutral.
- **Retrieval failures apply here exactly as they do in RAG.** A relevant memory that isn't retrieved is functionally the same as never having remembered it at all.

## Production considerations

- **Memory needs the same lifecycle management as any other stored data** — updating, expiring, and correcting entries, not just accumulating them indefinitely.
- **Privacy and retention policy apply directly to stored memories about real users** — what gets remembered, for how long, and who can access it are real product and compliance decisions, not implementation details.
- **Memory retrieval adds the same latency and token cost as any retrieval step**, per [Context Window](../llms/context-window.md) — worth budgeting for explicitly rather than treating memory as free context.

## Common mistakes

- **Storing every interaction as memory indiscriminately**, producing a noisy store where genuinely useful facts are hard to retrieve reliably among irrelevant ones.
- **Never updating or expiring old memories**, letting outdated facts silently mislead the agent long after they stopped being true.
- **Treating memory retrieval as guaranteed to succeed**, without the same evaluation discipline [Evaluation](../rag/evaluation.md) already established for RAG retrieval generally.

## Interview questions

### Basic

- What's the difference between a context window and agent memory?
- What are episodic, semantic, and procedural memory, and how do they differ?

### Intermediate

- Why is agent memory typically implemented using the same retrieval mechanism as RAG?
- What real risk does a stale or outdated memory pose to an agent's behavior?

### Advanced

- Design a memory system for a long-running personal assistant. What would you store, how would you decide what's worth remembering, and how would you handle outdated memories?
- How would you evaluate whether an agent's memory retrieval is actually helping versus introducing irrelevant or incorrect context?
