# Self-Supervised Learning

See [Types of Machine Learning](types-of-machine-learning.md) for where self-supervised learning fits among the other categories, and [Semi-Supervised Learning](semi-supervised-learning.md) for the adjacent technique this one is often confused with.

## What is it?

Self-supervised learning is a technique where a model generates its own labels directly from unlabeled data, by hiding part of the input and training the model to predict the hidden part — turning an unlabeled dataset into a supervised-style training signal without any human labeling at all.

Continuing the "answer key" framing from [Types of Machine Learning](types-of-machine-learning.md): self-supervised learning is a model writing its own answer key by hiding part of a question it already has the full answer to, then testing itself on the hidden part. No human ever writes a label; the data itself supplies both the question and the answer.

## Why does it exist?

Self-supervised learning is exactly what underpins modern LLM pretraining — the "next token prediction" objective behind [Transformers](../llms/transformers.md) and [Attention](../llms/attention.md) is a self-supervised task: given a huge corpus of raw, unlabeled text, hide the next word and have the model predict it, over and over, across billions of examples no human ever labeled. This exists because supervised learning could never scale to that much labeled data — nobody labeled the internet — and self-supervised learning is exactly the trick that sidesteps needing to.

**Self-supervised vs. semi-supervised — worth being precise about, since they sound similar but solve different problems.** [Semi-Supervised Learning](semi-supervised-learning.md) still needs *some* real human labels, just a small number, and uses unlabeled data to extend their reach. Self-supervised learning needs *no* human labels at all — the labels are extracted automatically from the data's own structure: the next word, a masked-out patch of an image. Self-supervised pretraining is typically followed by a smaller supervised fine-tuning step on real labels for a specific downstream task — the self-supervised phase builds a general-purpose representation, and a much smaller amount of real supervision then specializes it.

## How does it work?

Define a **pretext task** that can be automatically constructed from raw data with no human involvement: predicting the next token in a sequence, predicting a masked-out word from its surrounding context, or predicting whether two sentences originally followed each other in a document. Train a model on this automatically-generated task at massive scale, since the "labels" cost nothing to produce. The resulting model has learned a general-purpose representation that can then be adapted to specific downstream tasks with far less labeled data than training from scratch would need.

## Example

The simplest possible pretext task — predict the next word — with the "labels" extracted directly from raw text, no human annotation anywhere:

```python
from collections import defaultdict, Counter

corpus = """
the cat sat on the mat
the dog sat on the rug
the cat chased the mouse
the dog chased the cat
the mouse ran under the mat
"""

words = corpus.split()

# The pretext task: predict the next word from the current one.
# Every (word, next_word) pair is a free "label" pulled straight from the text.
bigram_counts = defaultdict(Counter)
for w1, w2 in zip(words, words[1:]):
    bigram_counts[w1][w2] += 1

def predict_next(word):
    return bigram_counts[word].most_common(1)[0][0]

print(predict_next("sat"))  # -> "on"
print(predict_next("the"))  # -> "cat"
```

`"sat" -> "on"` is learned from two occurrences in the corpus, both followed by "on" — a perfectly confident prediction, and nobody ever told the model "on" was the correct next word. This bigram counter is a deliberately minimal stand-in for the mechanism: a real LLM's next-token prediction is the same pretext task — predict what comes next from what came before — just learned by a Transformer over a corpus billions of times larger, producing far richer representations than word-frequency counts ever could.

## Where is it used?

Pretraining large language models on raw text (the entire basis of [Transformers](../llms/transformers.md)), pretraining vision models on unlabeled images (predicting a masked-out image patch), and any domain with abundant raw data but limited labels, where a general-purpose pretrained representation can then be fine-tuned on a much smaller labeled set for a specific task.

## Advantages

- **Scales to effectively unlimited data**, since the "labels" are generated automatically and cost nothing to produce.
- **Learns general-purpose representations** that transfer to many downstream tasks, rather than being specialized to one from the start.
- **Removes labeling as the bottleneck entirely**, unlike [Semi-Supervised Learning](semi-supervised-learning.md), which still needs some real labels to begin with.

## Limitations

- **The pretext task has to actually teach something useful for the real downstream task**, or the resulting representation transfers poorly no matter how much data it was trained on.
- **Training at the scale self-supervised learning is built for requires enormous compute** — the entire reason it works for LLMs is also the reason training one from scratch is far beyond most teams' resources.
- **A toy example like the one above is illustrative, not representative of scale.** The real power of the technique only shows up with far more data and a far more expressive model than a bigram counter.

## Production considerations

- **Fine-tuning a self-supervised pretrained model still needs the same evaluation discipline** as any other model, per [ML Workflow](ml-workflow.md) — a strong pretrained representation doesn't guarantee the fine-tuned result is correct for the specific task.
- **The pretext task's blind spots become the model's blind spots.** Next-token prediction learns a great deal about language, but nothing it wasn't exposed to in training data — the same distributional limitation already covered in [Embeddings](../embeddings/embeddings.md).
- **Reusing a pretrained model means inheriting its training data's biases and gaps**, not just its capabilities — worth auditing before fine-tuning it for a sensitive downstream task.

## Common mistakes

- **Assuming a pretrained self-supervised model needs no further evaluation** once fine-tuned, rather than validating it the same way any other model would be.
- **Confusing self-supervised learning with semi-supervised learning**, when they solve different problems: one needs zero labels, the other needs a small number.
- **Underestimating the scale gap** between illustrative examples (like the bigram counter above) and what actually makes self-supervised learning powerful in practice.

## Interview questions

### Basic

- What is a "pretext task" in self-supervised learning?
- How is self-supervised learning different from semi-supervised learning?

### Intermediate

- Why couldn't LLMs be trained the way they are today using purely supervised learning?
- Why does a self-supervised model typically get fine-tuned afterward, rather than used directly?

### Advanced

- What makes a pretext task "good" — likely to produce a representation that transfers well to real downstream tasks?
- A self-supervised pretrained model performs poorly after fine-tuning on a specific downstream task. How would you determine whether the problem is the pretraining, the fine-tuning data, or a mismatch between the two?
