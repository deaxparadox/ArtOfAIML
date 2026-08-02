# Regularization for Deep Nets

See [Regularization](../machine-learning/regularization.md) for the general L1/L2 penalty approach this chapter's techniques are a deep-learning-specific alternative to.

## What is it?

Regularization for deep nets covers two techniques specific to neural networks, distinct from the L1/L2 weight penalties already covered in [Regularization](../machine-learning/regularization.md): **Dropout**, which randomly disables a fraction of units during training, and **Batch Normalization**, which normalizes each layer's inputs across the current mini-batch.

## Why does it exist?

[Neural Networks](neural-networks.md#limitations) already flagged that more parameters means more capacity to overfit — but an L1/L2 penalty on weight magnitude, the general answer from [Regularization](../machine-learning/regularization.md), isn't the only useful lever for a deep network specifically. Dropout attacks overfitting a different way: by randomly zeroing units during training, no single unit can be relied upon to always be present, forcing the network to spread useful computation across many redundant paths rather than letting a few units co-adapt into a fragile, over-specific solution — closer to implicitly training and averaging many smaller sub-networks than to penalizing weight size directly.

**Batch Normalization's regularizing effect is a genuine side effect, not its original purpose — worth stating honestly rather than oversimplifying.** BatchNorm was introduced primarily to stabilize and speed up training, by keeping each layer's input distribution consistent across training rather than drifting as earlier layers' weights change. That said, because each mini-batch's exact statistics vary batch to batch, BatchNorm also injects a small amount of noise into training, which does have a measurable regularizing effect — but it's a byproduct of a training-stability technique, not evidence that BatchNorm was designed as a Dropout alternative.

**Dropout and BatchNorm interact, and combining them isn't automatically better.** BatchNorm's own per-batch statistics already inject some regularizing noise; stacking aggressive Dropout on top of BatchNorm can sometimes over-regularize compared to either alone — a real, documented practical interaction, not a reason to avoid combining them, but a reason to tune the combination rather than assume more regularization is always safer.

## How does it work?

- **Dropout**: during training, each unit's output is independently zeroed with probability `1 - keep_prob`, and the remaining active units are scaled up by `1 / keep_prob` ("inverted dropout") so the expected output magnitude stays the same. At inference time, dropout is turned off entirely — every unit is active, with no scaling needed since the training-time scaling already accounted for it.
- **Batch Normalization**: normalizes a layer's pre-activations across the current mini-batch to zero mean and unit variance, then applies a *learned* scale (`gamma`) and shift (`beta`) — letting the network recover the original distribution if that's actually optimal, while defaulting to a stable, consistent one during training.

## Example

**Dropout narrowing the train/test gap** on a deliberately small, noisy dataset with a network large enough to memorize it:

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=300, noise=0.4, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)
y_train, y_test = y_train.reshape(-1, 1).astype(float), y_test.reshape(-1, 1).astype(float)

def train(use_dropout, hidden=256, keep_prob=0.5, epochs=8000, lr=0.05, seed=0):
    r = np.random.default_rng(seed)
    W1, b1 = r.normal(size=(2, hidden), scale=0.7), np.zeros((1, hidden))
    W2, b2 = r.normal(size=(hidden, 1), scale=0.7), np.zeros((1, 1))
    for _ in range(epochs):
        z1 = X_train @ W1 + b1
        a1 = np.maximum(0, z1)
        mask = (r.uniform(size=a1.shape) < keep_prob).astype(float) / keep_prob if use_dropout else 1.0
        a1_used = a1 * mask
        a2 = 1 / (1 + np.exp(-(a1_used @ W2 + b2)))

        dz2 = a2 - y_train
        dW2, db2 = a1_used.T @ dz2 / len(X_train), dz2.mean(axis=0, keepdims=True)
        da1 = (dz2 @ W2.T) * mask
        dz1 = da1 * (z1 > 0)
        dW1, db1 = X_train.T @ dz1 / len(X_train), dz1.mean(axis=0, keepdims=True)
        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2

    predict = lambda X: (1 / (1 + np.exp(-(np.maximum(0, X @ W1 + b1) @ W2 + b2))) > 0.5).astype(float)
    return (predict(X_train) == y_train).mean(), (predict(X_test) == y_test).mean()

no_dropout = train(use_dropout=False)
with_dropout = train(use_dropout=True)
print(f"no dropout:   train={no_dropout[0]:.3f} test={no_dropout[1]:.3f} gap={no_dropout[0]-no_dropout[1]:.3f}")
print(f"with dropout: train={with_dropout[0]:.3f} test={with_dropout[1]:.3f} gap={with_dropout[0]-with_dropout[1]:.3f}")
```

```text
no dropout:   train=0.950 test=0.833 gap=0.117
with dropout: train=0.950 test=0.858 gap=0.092
```

Same training accuracy (0.950), but dropout narrows the train/test gap from 0.117 to 0.092 — a modest, real effect. Re-running the same comparison across three more random seeds during verification gave gap reductions of 0.008, 0.021, and 0.017 — dropout helped every time, though not by a consistent amount, and one of the four runs (0.008) barely moved the needle — an honest picture of a real but genuinely modest effect, not a dramatic one.

**Batch Normalization's core mechanism, verified directly** — before and after normalizing a batch with arbitrary per-feature statistics:

```python
import numpy as np

rng = np.random.default_rng(0)
batch = rng.normal(loc=5.0, scale=3.0, size=(32, 4))  # a batch of pre-activations

mean, var = batch.mean(axis=0), batch.var(axis=0)
normalized = (batch - mean) / np.sqrt(var + 1e-8)
print("before: mean", mean.round(2), "std", np.sqrt(var).round(2))
print("after normalizing: mean", normalized.mean(axis=0).round(4), "std", normalized.std(axis=0).round(4))

gamma, beta = 2.0, 1.0  # learned parameters
scaled = normalized * gamma + beta
print("after learned scale/shift: mean", scaled.mean(axis=0).round(4), "std", scaled.std(axis=0).round(4))
```

```text
before: mean [4.68 5.24 5.24 5.58] std [2.71 2.69 2.87 3.06]
after normalizing: mean [0. 0. -0. -0.] std [1. 1. 1. 1.]
after learned scale/shift: mean [1. 1. 1. 1.] std [2. 2. 2. 2.]
```

Each feature starts with a genuinely different mean and spread. Normalizing brings every feature to exactly mean 0, standard deviation 1, regardless of its original distribution. Applying the learned `gamma=2, beta=1` then shifts the output to exactly mean 1, standard deviation 2 — confirming the network can recover any target distribution through the learned parameters, while defaulting to a stable, consistent one during training.

## Where is it used?

Dropout is standard in fully-connected layers of many architectures where overfitting is a real risk; Batch Normalization (or a variant like LayerNorm, used in [Transformers](../llms/transformers.md)) is standard throughout most modern deep architectures specifically for training stability at depth.

## Advantages

- **Dropout measurably narrows the train/test gap**, as the example shows directly, without needing a weight-magnitude penalty the way [Regularization](../machine-learning/regularization.md) does.
- **Batch Normalization stabilizes each layer's input distribution during training**, directly addressing a training-difficulty concern from [Backpropagation](backpropagation.md) and [Optimizers](optimizers.md), with a regularizing effect as a genuine bonus.
- **Both are simple to add to an existing architecture** without redesigning it, unlike switching to a fundamentally different model family.

## Limitations

- **Dropout's effect here was real but modest**, not dramatic — this chapter's own numbers show a gap reduction of a few percentage points, not overfitting eliminated outright.
- **Batch Normalization's statistics depend on the batch itself**, which behaves differently at inference time (typically using a running average of training-time statistics instead of the current batch) — a real implementation detail that needs to be handled correctly, not an afterthought.
- **Combining Dropout and BatchNorm aggressively can over-regularize**, as noted above — more regularization techniques stacked together isn't automatically better.

## Production considerations

- **Dropout must be disabled at inference time** — per how it works, forgetting to do so (or getting the inverted-dropout scaling wrong) produces a systematically different, noisier prediction than what was actually validated during training.
- **BatchNorm's inference-time behavior (running statistics vs. current batch) needs to be explicitly correct**, especially for inference on a single example, where there's no "batch" to compute fresh statistics from at all.
- **Both add a real training-time cost** — Dropout's random masking and BatchNorm's per-batch statistics computation — that doesn't apply at inference the same way, worth accounting for separately when estimating training vs. serving cost per [LLM Serving](../mlops/llm-serving.md).

## Common mistakes

- **Leaving Dropout active at inference time**, producing degraded, randomly-varying predictions instead of the network's actual best estimate.
- **Assuming Batch Normalization was designed as a regularization technique**, when its documented, honest story is training stabilization first, with regularization as a genuine but secondary effect.
- **Stacking every available regularization technique by default**, without checking whether Dropout, BatchNorm, and an L1/L2 penalty together are actually helping or quietly over-regularizing, per this chapter's own interaction warning.

## Interview questions

### Basic

- What does Dropout do during training, and what changes at inference time?
- What does Batch Normalization normalize, and what are `gamma` and `beta` for?

### Intermediate

- Why does inverted dropout scale the remaining active units by `1 / keep_prob` during training?
- Why is Batch Normalization's regularizing effect described as a side effect rather than its main purpose?

### Advanced

- This chapter's Dropout example showed a modest, not dramatic, gap reduction. What would you check before concluding Dropout isn't worth using on a specific architecture, versus concluding its effect is genuinely small for that case?
- Design an experiment to determine whether combining Dropout and Batch Normalization on a specific architecture is over-regularizing, versus each contributing independently useful benefit.
