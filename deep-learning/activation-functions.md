# Activation Functions

See [Neural Networks](neural-networks.md) for why a non-linear activation is needed at all, and [Backpropagation](backpropagation.md) for the mechanism whose behavior this chapter's choice of function directly shapes.

## What is it?

An activation function is the non-linear function applied to each neuron's pre-activation value (`X @ W + b`) before passing it to the next layer. [Neural Networks](neural-networks.md#why-does-it-exist) already established that *some* non-linearity is what gives a multi-layer network representational power a single linear layer lacks — this chapter covers *which* non-linearity, and why that choice has a large, measurable effect on how well a network actually trains.

## Why does it exist?

Any non-linear function technically satisfies what [Neural Networks](neural-networks.md) requires for a hidden layer to add representational power — but [Backpropagation](backpropagation.md#how-does-it-work) showed that training multiplies gradients backward through every layer, one activation function's derivative at a time. If that derivative is small, repeatedly multiplying by it across many layers shrinks the gradient toward zero exponentially fast — the **vanishing gradient** problem — leaving early layers in a deep network receiving a gradient too small to learn from at all. The specific choice of activation function directly determines whether this happens.

**Sigmoid/tanh vs. ReLU — a real, measurable trade-off, not a stylistic preference.** Sigmoid and tanh squash their input into a bounded range (`(0, 1)` and `(-1, 1)` respectively) with a smooth, S-shaped curve — but their derivative is small everywhere and shrinks toward zero for any large-magnitude input, which is exactly what causes vanishing gradients across many layers. ReLU (`max(0, z)`) has a derivative of exactly 1 for any positive input — it doesn't shrink at all, letting gradients propagate through many layers without vanishing — but it introduces a different, real failure mode: a unit whose pre-activation is always negative outputs exactly zero and receives exactly zero gradient, forever — the **dying ReLU** problem, covered directly in this chapter's example.

## How does it work?

- **Sigmoid** (`1 / (1 + e^-z)`): output in `(0, 1)`, derivative is at most 0.25 (at `z=0`) and shrinks toward 0 as `|z|` grows. Still used at an output layer needing a probability, as [Supervised Learning](../machine-learning/supervised-learning.md#how-does-it-work) already covered for Logistic Regression.
- **Tanh** (`(e^z - e^-z) / (e^z + e^-z)`): output in `(-1, 1)`, derivative is at most 1 (at `z=0`) — better than sigmoid, but still shrinks toward 0 for large `|z|`, so it still vanishes across many layers, just more slowly.
- **ReLU** (`max(0, z)`): derivative is exactly 1 for `z > 0`, exactly 0 for `z < 0`. Doesn't shrink for positive inputs regardless of how deep the network is — the default choice for hidden layers in most modern architectures specifically because of this.
- **Leaky ReLU** and other variants give negative inputs a small non-zero slope instead of exactly zero, directly trading away some of ReLU's simplicity to avoid the dying-unit failure mode.

## Example

**Vanishing gradients, measured directly**: the same random 20-layer network, once with sigmoid activations and once with ReLU, comparing the gradient magnitude that actually reaches the very first layer.

```python
import numpy as np

rng = np.random.default_rng(0)
n_layers, width = 20, 16
sigmoid = lambda z: 1 / (1 + np.exp(-z))

def run(activation):
    x = rng.normal(size=(1, width))
    weights = [rng.normal(size=(width, width), scale=0.5) for _ in range(n_layers)]
    a, zs, activations = x, [], [x]
    for W in weights:
        z = a @ W
        zs.append(z)
        a = sigmoid(z) if activation == "sigmoid" else np.maximum(0, z)
        activations.append(a)

    grad = np.ones_like(activations[-1])
    for i in reversed(range(n_layers)):
        local_grad = activations[i+1] * (1 - activations[i+1]) if activation == "sigmoid" else (zs[i] > 0).astype(float)
        grad = (grad * local_grad) @ weights[i].T
    return np.linalg.norm(grad)

print("gradient reaching layer 1, sigmoid network:", run("sigmoid"))
print("gradient reaching layer 1, relu network:", run("relu"))
```

```text
gradient reaching layer 1, sigmoid network: 2.83e-09
gradient reaching layer 1, relu network:     5661.70
```

The sigmoid network's gradient at the first layer has shrunk to essentially zero after passing back through 20 layers — the first layer would receive almost no learning signal at all. The ReLU network's gradient reaching the same layer is still large — over a trillion times bigger — because ReLU's derivative doesn't shrink the gradient at every step the way sigmoid's does. Same depth, same random weights, only the activation function changed.

**Dying ReLU, measured directly**: a single unit whose pre-activation is pushed permanently negative by a large negative bias.

```python
import numpy as np

rng = np.random.default_rng(1)
X = rng.normal(size=(100, 3))
W = rng.normal(size=3)
b = -10.0  # strongly negative bias

z = X @ W + b
print("fraction of 100 inputs where this unit ever activates:", (z > 0).mean())
print("fraction where its gradient is exactly zero:", ((z > 0).astype(float) == 0).mean())
```

```text
fraction of 100 inputs where this unit ever activates: 0.0
fraction where its gradient is exactly zero: 1.0
```

Across all 100 inputs, this unit's pre-activation never crosses zero — it outputs exactly 0 every time, and its gradient is exactly 0 every time. Under plain gradient descent, a unit in this state can never recover: its incoming weights never receive a non-zero gradient to update them, regardless of what the rest of the network learns.

## Where is it used?

Every hidden layer in every network architecture in this handbook — ReLU or a variant is the default for hidden layers in most modern networks including [CNNs](cnns.md) and Transformers, while sigmoid remains standard specifically at an output layer producing a probability, and tanh still appears inside LSTM gates (covered in [RNNs / LSTMs](rnns-lstms.md)).

## Advantages

- **ReLU avoids vanishing gradients across depth**, as the example shows directly — a twelve-order-of-magnitude difference between the sigmoid and ReLU networks' gradients reaching the first layer, from the activation function alone, with nothing else about the architecture changed.
- **ReLU is computationally cheap** — a single comparison against zero, compared to sigmoid/tanh's exponential computation, at every unit in every layer.
- **Sigmoid still directly produces a valid probability**, exactly the property [Supervised Learning](../machine-learning/supervised-learning.md) relies on for Logistic Regression's output layer.

## Limitations

- **ReLU units can die permanently**, as this chapter's example shows concretely — a large negative bias, or an unlucky combination of weights, can leave a unit outputting zero forever with no way to recover under plain gradient descent.
- **Sigmoid and tanh genuinely vanish across depth**, as the first example demonstrates — not a minor slowdown, but a gradient shrunk by nine orders of magnitude across only 20 layers.
- **No single activation function is correct everywhere** — the right choice depends on the layer's role (hidden vs. output) and the architecture, not a single default applied blindly throughout.

## Production considerations

- **A stalled or `NaN` loss during training is often traceable to activation choice** — per [Backpropagation](backpropagation.md#production-considerations)'s own note that a `NaN` loss is usually a gradient problem, checking for dead ReLU units or vanishing gradients is a standard first diagnostic step.
- **Monitoring the fraction of dead units in a ReLU network during training** catches the dying-ReLU problem while it's still fixable (lower learning rate, different initialization, a leaky variant), rather than after most of a layer has already gone silent.
- **Changing an activation function changes the entire training dynamics**, not just the forward-pass output — a swap made casually can require re-tuning learning rate and initialization alongside it.

## Common mistakes

- **Using sigmoid or tanh throughout a deep hidden stack** by default, then being confused why early layers don't seem to learn — exactly the vanishing-gradient effect this chapter's example measures directly.
- **Initializing weights or biases in a way that pushes many ReLU units permanently negative** from the start, effectively wasting a large fraction of the network's capacity before training even begins.
- **Assuming a "modern" activation function is always the safe default everywhere** — an output layer producing a probability still needs sigmoid (binary) or softmax (multi-class), not ReLU.

## Interview questions

### Basic

- What is the vanishing gradient problem, and which activation functions are prone to it?
- What does it mean for a ReLU unit to "die," and why can't plain gradient descent recover it?

### Intermediate

- Why does ReLU's derivative not shrink the gradient the way sigmoid's does, and why does that matter specifically in a deep network?
- Why is sigmoid still used at an output layer even though it's avoided in hidden layers of deep networks?

### Advanced

- This chapter's example measured a gradient shrinking from a normal magnitude to 2.8e-9 across 20 sigmoid layers. At what depth would you expect this to become a practical training problem, and what would you check to confirm vanishing gradients are the actual cause versus something else?
- Design a strategy to detect and recover from dying ReLU units during training, without switching the entire network to a different activation function.
