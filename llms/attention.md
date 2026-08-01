# Attention

See [Transformers](transformers.md) for why this mechanism was needed in the first place.

## What is it?

Attention is the mechanism that lets each position in a sequence decide, numerically, how much to weigh every other position's information when building its own updated representation. For each position, it produces a weighted blend of every position's "value," where the weights are computed based on how relevant each other position is to this one.

Continuing directly from [Transformers](transformers.md)'s "every word laid out on a table" analogy: attention is the specific rule for how much each word listens to every other word. Each word asks a question (its **Query** — what context am I looking for), every word, including itself, offers an answer key (its **Key** — what do I represent, for others to match against) and a piece of content (its **Value** — what information do I actually carry), and the result is a blend of everyone's content, weighted by how well their key matched the asking word's query.

## Why does it exist?

[Transformers](transformers.md) explained why a way to connect any two positions directly, regardless of distance, was needed. Attention is the actual mechanism that does that connecting. A fixed rule — like averaging every position equally — would be fine if every word in a sentence mattered equally to every other word, but real language isn't like that: in "the cat, which was hungry, ate the fish," the word "ate" needs to connect strongly to "cat" and "fish," and weakly to "which" and "was." Attention exists to let that weighting be learned and computed dynamically, per input, rather than fixed in advance.

**The Query/Key/Value split is the core mechanism, not a side detail.** Separating "what to match against" (Key) from "what to actually pass along" (Value) lets a model learn two different things about the same token independently: how relevant it should seem to others asking about it, and what content it actually contributes once selected. Collapsing the two — using the same vector for both matching and content — would tie a token's relevance directly to its content, a real restriction attention specifically avoids.

## How does it work?

Standard **scaled dot-product attention**:

1. Every position produces a Query, Key, and Value vector, via three separate learned linear projections of its embedding.
2. Raw attention scores are computed as `Query · Key^T` for every pair of positions — how much position `i`'s query matches position `j`'s key.
3. Scores are scaled by `1/sqrt(d_k)` (the key dimension) to keep values in a numerically stable range before the next step.
4. **Softmax** is applied across each row, turning raw scores into a probability distribution — how much of position `i`'s attention goes to each other position, summing to exactly 1.
5. Those weights multiply the Value vectors and sum, producing the final output for each position: a weighted blend of everyone's value, weighted by how relevant they were.

## Example

A real, from-scratch implementation over four toy token embeddings — two of them deliberately made nearly identical, to check whether the mechanism actually notices:

```python
import numpy as np

rng = np.random.default_rng(0)
d_model, d_k = 8, 8

tok_a = rng.normal(size=d_model) * 1.3
tok_b = rng.normal(size=d_model) * 1.3
tok_c = rng.normal(size=d_model) * 1.3
embeddings = np.array([
    tok_a,
    tok_b,
    tok_a + rng.normal(scale=0.05, size=d_model),  # nearly identical to token 0
    tok_c,
])

W_q = rng.normal(size=(d_model, d_k))
W_k = rng.normal(size=(d_model, d_k))
W_v = rng.normal(size=(d_model, d_k))

Q, K, V = embeddings @ W_q, embeddings @ W_k, embeddings @ W_v
scores = Q @ K.T / np.sqrt(d_k)

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

weights = softmax(scores)
print(weights.sum(axis=1))       # -> [1. 1. 1. 1.]
print(np.round(weights[0], 3))   # -> [0.414 0.    0.328 0.258]
```

Every row sums to exactly 1, confirming attention weights form a real probability distribution over the other positions. Token 0's own row shows the actual point: it attends most to itself (0.414), second-most to token 2 — its near-duplicate — at 0.328, then to the unrelated token 3 at 0.258, and almost nothing to token 1, the least related of the three. `W_q`, `W_k`, and `W_v` here are random, untrained matrices, so this isn't a trained model's learned attention pattern — but it shows the mechanism itself doing exactly what it's supposed to: routing more weight toward the token that's actually similar.

## Where is it used?

Every layer of every Transformer-based model — this is the core mechanism, not an optional add-on. Variants of it (multi-head attention, running several of these in parallel with different learned projections; cross-attention, where queries come from one sequence and keys/values from another) show up throughout modern LLM architecture and beyond it, in retrieval and multimodal systems.

## Advantages

- **Learns which positions matter to which other positions dynamically**, rather than relying on a fixed rule like a simple average.
- **Connects any two positions in a single step**, regardless of how far apart they are in the sequence.
- **The Query/Key/Value split lets a model learn relevance and content as two separate things** about the same token.

## Limitations

- **Computing scores between every pair of positions is exactly what makes attention scale quadratically with sequence length**, as [Transformers](transformers.md) already covered.
- **Attention weights are a weaker explanation than they're often given credit for.** A high weight shows the mechanism routed information from one position to another — not that the connection is meaningful or correct in a human sense.
- **This example shows the mechanism working correctly in isolation.** A real, useful attention pattern only emerges after training on real data, which this small example, using random untrained projections, deliberately doesn't do.

## Production considerations

- **Multi-head attention multiplies the same quadratic cost by the number of heads** — a real, direct driver of inference cost and latency, not just an architectural detail.
- **Caching previously computed Key and Value vectors** (so they don't need recomputing for every new token generated) is a standard, load-bearing optimization in production LLM serving — without it, generating each new token would redo attention over the entire sequence from scratch.
- **Attention weights are sometimes surfaced to end users as an "explanation" for a model's output.** Given the limitation above, that framing needs care — it's a real signal about information flow, not a guarantee of a causally meaningful connection.

## Common mistakes

- **Treating an attention weight as proof a model "understood" a specific relationship**, rather than as evidence that information flowed from one position to another during that specific computation.
- **Forgetting that multi-head attention multiplies attention's already-quadratic cost by the number of heads** when estimating a model's compute or latency budget.
- **Assuming the Query/Key/Value split is a minor implementation detail**, rather than the specific design choice that lets a model separate "how relevant is this" from "what does this actually contribute."

## Interview questions

### Basic

- What are Query, Key, and Value, and what role does each play?
- Why is softmax applied to attention scores before they're used to weight the Values?

### Intermediate

- Why are Query and Key kept separate from Value, instead of using the same vector for all three roles?
- What does it mean for attention weights across a row to sum to 1?

### Advanced

- Why does caching Key and Value vectors matter for the cost of generating long sequences in production?
- An attention weight between two tokens is high. What can you actually conclude from that, and what can't you conclude?
