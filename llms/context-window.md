# Context Window / Context Length

See [Attention](attention.md) for the quadratic cost this chapter's limit is a direct consequence of.

## What is it?

The context window (or context length) is the maximum number of tokens a model can process in a single request — including both the input (prompt, retrieved context, conversation history) and the output the model generates. Once that limit is reached, older or excess content has to be dropped, summarized, or otherwise managed.

If tokens are the units a model actually consumes, as established in [Tokenization](../nlp/tokenization.md), the context window is the size of the plate the model can hold at once. Everything relevant to a request — the question, any retrieved documents, prior conversation turns — has to fit on that plate; anything that doesn't isn't seen by the model at all, no matter how relevant it might have been.

## Why does it exist?

[Transformers](transformers.md) established that attention's computational cost grows quadratically with sequence length — that's not an incidental detail, it's the direct, structural reason a context window has a hard limit at all. A model can't simply process an unlimited amount of input, because doing so would require unlimited compute and memory, growing quadratically rather than linearly with how much text is fed in. The context window is where that trade-off becomes concrete and visible to anyone building with an LLM: a fixed, stated number of tokens describing exactly how much a specific model can consider at once.

**What to do when content exceeds the window is a real, constant engineering decision.** Truncating the oldest conversation history is simple, but can lose important early context — a user's stated preference at the start of a long conversation. Summarizing older content before dropping it preserves the gist at the cost of detail and an extra model call to do the summarizing. Retrieving only the most relevant pieces of a larger collection instead of feeding all of it — exactly what [RAG](../rag/what-is-rag.md) exists to do — avoids the problem entirely by being selective about what gets included in the first place, rather than reactively managing overflow after the fact.

## How does it work?

Every request's total token count — input plus expected output — has to stay under the model's stated context window. As context grows within that limit, cost and latency grow too, per [Transformers](transformers.md)' quadratic attention cost and [Tokenization](../nlp/tokenization.md)'s own cost-per-token point — staying under the limit doesn't mean staying free of its cost. Some model families support much larger context windows than others, but a larger window is not free capability — it's a direct cost and latency trade-off, exactly as [Transformers](transformers.md#production-considerations) already flagged.

## Example

A conversation growing past a hypothetical 60-token context window, and a simple truncation strategy handling it:

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

conversation = [
    "What is the return policy for electronics?",
    "Electronics can be returned within 15 days if unopened.",
    "What about clothing items?",
    "Clothing can be returned within 30 days with tags attached.",
    "Do you offer store credit instead of refunds?",
    "Yes, store credit is available and does not expire.",
    "Can I return a gift without a receipt?",
    "Gift returns without a receipt are issued as store credit at the current price.",
]

def total_tokens(messages):
    return sum(len(enc.encode(m)) for m in messages)

for i in range(1, len(conversation) + 1):
    print(i, total_tokens(conversation[:i]))

context_window = 60
kept = conversation.copy()
while total_tokens(kept) > context_window:
    kept.pop(0)

print(len(conversation) - len(kept), "messages dropped")
print(total_tokens(kept), "tokens remaining")
```

```text
running totals: 8, 21, 26, 39, 48, 59, 68, 83
context window: 60
messages dropped: 3
remaining token count: 57
```

The conversation fits comfortably through message 6 (59 tokens), then crosses the 60-token limit at message 7. A simple oldest-first truncation strategy brings the total back to 57 tokens — under budget — but it does so by dropping the very first exchange, about the electronics return policy. If a later question in this conversation referred back to that policy, the model would have no memory of it at all; the truncation kept the conversation within budget at the direct cost of silently losing real, potentially relevant context.

## Where is it used?

Every conversational LLM application managing multi-turn history, every RAG system deciding how much retrieved content fits alongside a query, and any system with a hard limit on total prompt plus completion length.

## Advantages

- **Makes the trade-off explicit and quantifiable** — a fixed, stated number, rather than an ambiguous "how much context is too much."
- **Forces a deliberate strategy for overflow** — truncation, summarization, or retrieval — rather than an application silently failing or behaving unpredictably.
- **Scales cost and latency predictably with context length**, once the underlying quadratic relationship from [Attention](attention.md) is understood.

## Limitations

- **Any overflow-handling strategy loses something.** Truncation loses detail, summarization loses precision and adds cost, retrieval only works if the retrieval step itself is accurate.
- **A larger context window isn't a substitute for good retrieval or summarization.** It raises the ceiling, but doesn't remove the trade-off — it just moves where it becomes a problem, and at real added cost per [Attention](attention.md#production-considerations).
- **Token counting has to happen before the request, not after.** Discovering a request exceeded the context window only from an error response is a late, avoidable failure mode.

## Production considerations

- **Overflow strategy needs to be decided deliberately, not left to whatever a library happens to do by default** — silently truncating from the wrong end of a conversation, for instance, could drop a system instruction instead of old chat history.
- **Token budgets should be tracked proactively**, the same discipline already established in [Prompt Engineering](prompt-engineering.md) and [Chunking](../rag/chunking.md) — count before sending, not after a request fails.
- **Context window size differs across model versions and providers**, and swapping models can silently change how much context actually fits, breaking an application that assumed a specific limit.

## Common mistakes

- **Not proactively tracking token counts**, discovering the context window was exceeded only from a failed request instead of preventing it in advance.
- **Truncating from the wrong end of a conversation**, dropping a critical system instruction or early user preference instead of less relevant recent chatter.
- **Assuming a larger context window solves a retrieval or summarization problem**, when it only postpones the same trade-off to a larger scale, at real additional cost.

## Interview questions

### Basic

- What counts toward a model's context window: just the input, or the input and output together?
- Why does a context window have a hard limit at all?

### Intermediate

- What are the trade-offs between truncating, summarizing, and retrieving when content exceeds the context window?
- Why isn't a larger context window a free upgrade, even though it raises the ceiling on how much can be included?

### Advanced

- Design a strategy for managing a long-running conversation that needs to remember an early user preference indefinitely, without exceeding the context window on every subsequent turn.
- A production application starts failing after a model provider update. How might a change to the context window be involved, even if the code never changed?
