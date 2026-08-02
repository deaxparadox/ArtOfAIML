# LLM Serving

See [FastAPI](fastapi.md) for the general serving pattern this chapter specializes for LLM-scale inference.

## What is it?

LLM serving is the practice of running a large language model as a live service that handles inference requests efficiently at scale — distinct from serving a small classical model because an LLM's inference is far more compute-intensive per request, and serving many concurrent requests efficiently requires techniques specific to how LLMs actually generate text.

[FastAPI](fastapi.md) wraps a model behind an HTTP endpoint, and for a small classical model like the house-price model in that chapter, calling `.predict()` inside the endpoint is computationally almost free. An LLM's forward pass is not free at all — generating each token requires a full pass through billions of parameters, and generating a full response means doing that many times in sequence, once per output token. LLM serving frameworks exist specifically to make that per-request cost as efficient as possible, especially with many concurrent users.

## Why does it exist?

[FastAPI](fastapi.md#production-considerations) already named the bottleneck directly: "for an endpoint wrapping a large LLM call, that inference time, not FastAPI itself, is almost always the real bottleneck." LLM serving exists to actually address that bottleneck, with techniques a generic web framework has no way to provide on its own: batching multiple users' requests together so a GPU processes them more efficiently than one at a time, managing how the model's intermediate computation — the KV cache, already introduced in [Attention](../llms/attention.md#production-considerations) — is stored and reused across requests, and carefully managing GPU memory, since a large model's weights alone can consume most of a GPU's available memory.

**A generic web framework vs. a specialized LLM serving framework — a real decision depending on what's actually being served.** Wrapping a small classical model, or even calling a hosted LLM API from within a FastAPI endpoint, is exactly what FastAPI is built for — the LLM inference itself happens elsewhere, and FastAPI's job is just the surrounding request handling. Self-hosting and serving an LLM's actual weights directly is a fundamentally heavier workload, and that's specifically when a dedicated serving framework (vLLM, TGI) earns its complexity — techniques like continuous batching and efficient KV cache management can improve throughput dramatically over naively wrapping the model in a basic web framework.

## How does it work?

**Continuous batching** adds new requests to an in-flight batch as older ones finish generating, instead of waiting for a fixed batch to arrive before processing — keeping the GPU busy without wasted idle time. **KV cache management** reuses the key/value vectors from [Attention](../llms/attention.md) across generation steps so each new token doesn't require redoing attention over the entire sequence from scratch, now managed across many concurrent requests without exhausting GPU memory. **Quantization** reduces the numeric precision of a model's weights — from 32-bit down to 8-bit or 4-bit — to fit a larger model into less memory and run inference faster, at some cost to output precision.

## Example

A full GPU-based LLM serving benchmark isn't feasible to run here, but the core throughput principle batching exploits — processing many things together beats processing them one at a time on the same hardware — is real and directly verifiable, the same honest-analogy approach [Transformers](../llms/transformers.md#example) used for its own parallelization example:

```python
import numpy as np
import timeit

rng = np.random.default_rng(0)
requests = rng.normal(size=(500, 512))  # 500 "requests," standing in for a simplified model step
W = rng.normal(size=(512, 512))

def one_at_a_time():
    return np.array([req @ W for req in requests])

def batched():
    return requests @ W

t_seq = timeit.timeit(one_at_a_time, number=20) / 20
t_batch = timeit.timeit(batched, number=20) / 20
print(f"one-at-a-time: {t_seq*1000:.2f} ms")
print(f"batched:       {t_batch*1000:.2f} ms")
```

```text
one-at-a-time: 15.83 ms
batched:       1.80 ms
```

Both approaches compute the exact same result — verified identical — but processing all 500 requests together runs 8.8x faster than handling them one at a time. This isn't a real LLM serving benchmark; it's a simplified stand-in showing the exact structural principle continuous batching exploits: the same hardware does dramatically more useful work per unit time when requests are processed together rather than sequentially, whether that's a matrix multiply here or a Transformer's forward pass in a real serving engine.

## Where is it used?

Self-hosting an open-weight LLM for a production application, any system where inference cost or latency at scale matters enough to justify specialized serving infrastructure over a generic web framework, and GPU-constrained environments where memory efficiency directly determines how large a model can be served at all.

## Advantages

- **Dramatically improves throughput under concurrent load**, as the example shows directly — the same underlying principle, applied to real model inference instead of a single matrix multiply.
- **KV cache reuse avoids redundant computation** across the many sequential steps of generating one response, and across multiple concurrent requests.
- **Quantization lets larger models fit into available GPU memory**, trading some precision for the ability to serve a model that wouldn't otherwise fit at all.

## Limitations

- **Batching benefits depend on request patterns.** A serving system with very few concurrent requests sees little of the throughput gain this chapter's example demonstrates.
- **Quantization trades accuracy for memory and speed** — the right amount depends on how much output quality degradation is acceptable for the specific use case.
- **Specialized serving frameworks add real operational complexity** compared to a plain FastAPI endpoint calling a hosted API — worth it only when self-hosting is actually the right call in the first place.

## Production considerations

- **GPU memory is the hard constraint** most LLM serving decisions revolve around — model size, batch size, and KV cache size all compete for the same fixed memory budget.
- **Batching improves average throughput but can affect individual request latency** — a request arriving mid-batch may wait slightly longer than it would have alone, a real trade-off between overall efficiency and per-request responsiveness.
- **Self-hosting vs. a hosted API is itself a cost and operations decision**, not just a technical one — self-hosting can be cheaper at high, steady volume, but adds the infrastructure and expertise cost this chapter covers.

## Common mistakes

- **Serving a self-hosted LLM behind a plain web framework without batching or KV cache management**, leaving significant throughput on the table that a purpose-built serving engine would capture.
- **Applying aggressive quantization without checking its actual impact on output quality** for the specific task, rather than assuming the memory savings are free.
- **Choosing to self-host an LLM without comparing the real infrastructure cost against a hosted API's per-token pricing** at the actual expected volume.

## Interview questions

### Basic

- Why does serving an LLM require different techniques than serving a small classical model?
- What does continuous batching do, at a high level?

### Intermediate

- Why does batching improve throughput, and what did the example in this chapter verify about that?
- What does quantization trade away in exchange for reduced memory usage?

### Advanced

- How would you decide between self-hosting an LLM with a dedicated serving framework versus calling a hosted API, for a given expected request volume?
- Why can batching improve average throughput while potentially increasing latency for an individual request, and how would you balance that trade-off in a latency-sensitive application?
