# Transformers

## What is it?

A Transformer is a neural network architecture built around a mechanism called [attention](attention.md), designed to process sequences — most commonly, sequences of token embeddings — by letting every position in the sequence directly gather information from every other position, all at once, rather than one step at a time.

Think of an older, sequential architecture (like a recurrent neural network) as reading a book one word at a time, carrying forward only a running summary — by the last word, details from the first word have faded. A Transformer instead lays every word of the book on a table at once and lets each word directly look at every other word, however far away, to figure out what's relevant to it.

## Why does it exist?

Before Transformers (introduced in 2017's "Attention Is All You Need"), the standard architecture for sequence data was the recurrent neural network (RNN), which processes a sequence one element at a time, carrying a hidden state forward. This creates two structural problems. First, it's inherently sequential — position 50 can't be processed until position 49 is done, which prevents parallelizing computation across the sequence and makes training on large datasets slow. Second, information from early in a long sequence has to survive being repeatedly compressed into the same fixed-size hidden state at every step, so it tends to fade — a long-range dependency problem, where a sequential model struggles to connect a word to something much earlier in the same sequence.

The Transformer architecture solves both at once. Attention lets every position connect directly to every other position in a single step — no fading with distance, no waiting for a sequential chain — and because computing one position's output doesn't depend on having already computed another's, the entire sequence can be processed in parallel. That parallelizability is exactly what made training the very large models behind modern LLMs computationally feasible in the first place.

**This isn't really "when would you pick an RNN over a Transformer" in modern practice** — Transformers dominate text, and increasingly other modalities, specifically because of the parallelization advantage above. But the older approach wasn't simply worse; it made a genuine, different trade: an RNN's per-step compute and memory stay constant regardless of sequence length, while a Transformer's core attention computation scales quadratically with sequence length, since every position attends to every other position. That's exactly why processing very long sequences efficiently remains a real, unsolved engineering problem for Transformer-based models, not a footnote — a large share of current LLM research is specifically about making attention cheaper over long contexts.

## How does it work?

At a high level, without yet covering the attention computation itself (see [Attention](attention.md)):

1. Input tokens become embeddings, as covered in [Embeddings](../embeddings/embeddings.md).
2. Attention itself has no inherent sense of order — it treats a sequence as a set of positions, not a sequence — so positional information gets added to each embedding, letting the model tell "first token" from "fifth token" apart.
3. Multiple layers of attention, each followed by a small feed-forward network, are stacked on top of each other. Each layer lets every position gather more context from the rest of the sequence and refine its own representation.
4. The final layer's output at a given position is used to make a prediction — for a language model, typically "what token comes next."

## Example

A simplified, honest illustration of the specific property that motivates Transformers — not the actual attention mechanism itself, covered in [Attention](attention.md) — comparing a computation that must proceed one step at a time against one where every position's result depends only on the shared input, not on another position's output:

```python
import numpy as np
import timeit

n = 2000
x = np.random.default_rng(0).normal(size=(n, 32))  # a toy sequence of "embeddings"
W = np.random.default_rng(1).normal(size=(32, 32))

def sequential_style():
    # each step depends on the previous step's result - the RNN pattern
    state = np.zeros(32)
    outputs = []
    for i in range(n):
        state = np.tanh(x[i] @ W + state)
        outputs.append(state)
    return np.array(outputs)

def parallel_style():
    # each position depends only on the shared input - the property attention relies on
    return np.tanh(x @ W)

t_seq = timeit.timeit(sequential_style, number=20) / 20
t_par = timeit.timeit(parallel_style, number=20) / 20
print(f"sequential: {t_seq*1000:.2f} ms")
print(f"parallel:   {t_par*1000:.2f} ms")
```

```text
sequential: 7.51 ms
parallel:   1.16 ms
```

About 6.5x faster for the parallel version, on a comparable amount of arithmetic — the same kind of vectorization advantage already verified directly in [NumPy](../python-for-ml/numpy.md), just illustrating why the *structural* independence between positions, not carrying a dependency from one step to the next, is what makes an entire sequence computable at once rather than one token at a time. Real attention is more involved than this — computing how much each position should actually weigh every other position is what [Attention](attention.md) covers — but this independence is the property that makes that computation parallelizable in the first place.

## Where is it used?

The architecture behind essentially every modern large language model, and increasingly vision and audio models too — anywhere a sequence needs to be processed with long-range context and large-scale, parallelizable training.

## Advantages

- **Processes an entire sequence in parallel during training**, unlike a sequential architecture that must proceed one step at a time.
- **Connects any two positions directly, regardless of distance**, avoiding the long-range dependency fading that sequential architectures struggle with.
- **Scales remarkably well with more data and compute**, the main reason larger Transformer-based models have kept improving as they've grown.

## Limitations

- **Its core attention computation scales quadratically with sequence length** — doubling the input length roughly quadruples the attention computation, exactly why very long contexts are expensive.
- **Needs large amounts of data and compute to train well from scratch**, unlike some older architectures that could be trained reasonably on more limited data.
- **Positional information has to be explicitly added** rather than being inherent to the architecture, since attention itself has no built-in sense of sequence order.

## Production considerations

- **Inference cost and latency both grow with sequence length**, because of the same quadratic attention cost that affects training — a system's decision to support longer context windows is a direct cost and latency trade-off, not a free capability upgrade.
- **Batch size, sequence length, and available GPU memory interact directly**, since attention's memory usage also grows with sequence length — a model that runs fine on short inputs can run out of memory purely from being given a longer one.
- **Swapping to a newer or larger Transformer-based model isn't a drop-in replacement for cost and latency planning.** Larger models are typically slower and more expensive per request, even when the interface looks identical.

## Common mistakes

- **Assuming a longer context window is free or nearly free**, when the underlying attention computation cost grows quadratically with it.
- **Treating all Transformer-based models as interchangeable in terms of cost and latency**, without checking the model size and attention implementation details that affect both directly.
- **Conflating "Transformer" with "attention" as if they were the same thing.** The Transformer is the overall architecture; attention is the specific mechanism inside it, covered in depth in [Attention](attention.md).

## Interview questions

### Basic

- What problem with RNNs did the Transformer architecture solve?
- Why can a Transformer process a sequence in parallel, while an RNN cannot?

### Intermediate

- Why does attention need positional encoding, when an RNN doesn't need anything equivalent?
- What does it mean for attention's computational cost to scale quadratically with sequence length?

### Advanced

- Why is supporting a longer context window a real cost and latency trade-off for a production LLM system, not just a capability toggle?
- An RNN keeps its per-step memory and compute constant regardless of sequence length, while a Transformer doesn't. What real production scenario would that trade-off actually matter for?
