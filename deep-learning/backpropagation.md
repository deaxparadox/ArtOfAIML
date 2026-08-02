# Backpropagation

See [Neural Networks](neural-networks.md) for the layered architecture this chapter trains, and [Supervised Learning](../machine-learning/supervised-learning.md#example) for the gradient descent mechanism this chapter extends across multiple layers.

## What is it?

Backpropagation is the algorithm for computing the gradient of a network's loss with respect to every weight and bias in every layer, efficiently, by applying the chain rule backward from the output layer to the input layer — reusing shared intermediate computations rather than recomputing each weight's gradient independently from scratch.

[Supervised Learning](../machine-learning/supervised-learning.md#example) verified gradient descent directly for Linear and Logistic Regression, where the gradient of the loss with respect to the (single layer of) weights has a simple, direct formula. A multi-layer network's loss depends on an early layer's weights only indirectly — through every layer that comes after it — so computing that gradient means accounting for the entire chain of dependencies in between. Backpropagation is specifically how that chain is computed without it becoming computationally intractable as depth grows.

## Why does it exist?

[Neural Networks](neural-networks.md#how-does-it-work) established that training adjusts every layer's weights via gradient descent — but didn't say how the gradient for an early layer's weights is actually computed, given that changing those weights affects the loss only through everything downstream of them. Naively, the chain rule could be expanded separately for every individual weight, but this recomputes many of the same intermediate terms over and over — the cost would grow explosively with depth. Backpropagation exists to compute the gradient for *every* weight in the network in a single backward pass, reusing each layer's intermediate result for every weight that depends on it, at a total cost roughly proportional to just one additional forward pass.

**Reverse-mode vs. forward-mode differentiation — the real reason backprop is reverse, not forward.** A neural network's loss has many inputs (every weight) and one output (a single loss value). Reverse-mode differentiation (backpropagation) computes the gradient with respect to *all* inputs in one backward pass, at a cost independent of how many weights there are — exactly the situation a network with millions of weights and one loss value needs. Forward-mode differentiation would need a separate pass per input to get its gradient, intractable the moment there are more than a handful of parameters. This isn't a minor implementation choice; it's the specific reason training a network with millions of weights is computationally feasible at all.

## How does it work?

1. **Forward pass**: compute and cache every layer's pre-activation and post-activation values, exactly as in [Neural Networks](neural-networks.md#example) — these cached values are what the backward pass reuses.
2. Compute the loss at the output layer.
3. Compute the gradient of the loss with respect to the *output* layer's pre-activation — for sigmoid output with binary cross-entropy loss, this has a famously simple closed form: `prediction - true_label`.
4. **Propagate backward, layer by layer**: at each layer, use the chain rule to turn "gradient with respect to this layer's output" into "gradient with respect to this layer's weights and bias" (needed for the weight update) and "gradient with respect to the previous layer's output" (needed to keep propagating backward) — multiplying by the local activation function's derivative at each step.
5. Every weight and bias's gradient, once computed, feeds into a gradient descent update — Optimizers (its own upcoming chapter) covers what actually happens beyond the plain update rule already verified in [Supervised Learning](../machine-learning/supervised-learning.md#example).

## Example

**Gradient checking**: verifying a from-scratch backprop implementation against independently-computed numerical gradients — the standard way to confirm a manual backprop implementation is actually correct, not just plausible-looking.

```python
import numpy as np

rng = np.random.default_rng(0)
x = rng.normal(size=(1, 3))
y_true = np.array([[1.0]])
W1, b1 = rng.normal(size=(3, 4)), rng.normal(size=(1, 4))
W2, b2 = rng.normal(size=(4, 1)), rng.normal(size=(1, 1))

relu = lambda z: np.maximum(0, z)
relu_grad = lambda z: (z > 0).astype(float)
sigmoid = lambda z: 1 / (1 + np.exp(-z))

def forward(x, W1, b1, W2, b2):
    z1 = x @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

def loss_fn(a2, y_true):
    return -(y_true * np.log(a2) + (1 - y_true) * np.log(1 - a2)).sum()

z1, a1, z2, a2 = forward(x, W1, b1, W2, b2)

# manual backprop
dL_dz2 = a2 - y_true                     # sigmoid + cross-entropy's well-known simplification
dL_dW2 = a1.T @ dL_dz2
dL_dz1 = (dL_dz2 @ W2.T) * relu_grad(z1)
dL_dW1 = x.T @ dL_dz1

# numerical gradient via finite differences, for comparison
def numerical_grad(param, eps=1e-5):
    grad = np.zeros_like(param)
    for idx in np.ndindex(param.shape):
        orig = param[idx]
        param[idx] = orig + eps
        loss_plus = loss_fn(forward(x, W1, b1, W2, b2)[3], y_true)
        param[idx] = orig - eps
        loss_minus = loss_fn(forward(x, W1, b1, W2, b2)[3], y_true)
        param[idx] = orig
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
    return grad

print("max |analytic - numerical| in dW1:", np.max(np.abs(dL_dW1 - numerical_grad(W1))))
print("max |analytic - numerical| in dW2:", np.max(np.abs(dL_dW2 - numerical_grad(W2))))
```

```text
max |analytic - numerical| in dW1: 0.0
max |analytic - numerical| in dW2: 0.0
```

The hand-derived backpropagation gradients match independently-computed numerical gradients exactly, down to floating-point precision. This is exactly how a real backprop implementation gets verified in practice: not by "does the loss go down," which a subtly wrong gradient can still sometimes appear to do, but by directly comparing the analytic gradient against a numerical approximation that doesn't depend on the chain-rule derivation being correct at all.

## Where is it used?

Training every neural network architecture in this handbook and beyond — [Transformers](../llms/transformers.md), [Attention](../llms/attention.md), CNNs and RNNs / LSTMs (their own upcoming chapters) — all use backpropagation as the mechanism for computing gradients; what varies between architectures is the specific chain of operations being differentiated, not the underlying algorithm.

## Advantages

- **Computes every weight's gradient in one backward pass**, at a cost roughly proportional to a single additional forward pass, regardless of how many weights the network has.
- **Reuses shared intermediate computations**, avoiding the explosive redundant recomputation a naive per-weight chain-rule expansion would require.
- **Verifiable independently of the derivation**, as the example shows directly — gradient checking against numerical gradients confirms correctness without trusting the chain-rule algebra alone.

## Limitations

- **Requires every operation in the forward pass to be differentiable** (or have a defined subgradient, as ReLU does at zero) — an architecture built from a non-differentiable operation breaks the chain the algorithm depends on.
- **Gradients can vanish or explode across many layers**, an issue directly tied to which activation function is used — covered in depth in the next chapter, [Activation Functions](activation-functions.md).
- **Memory cost scales with network depth**, since every layer's forward-pass values need to be cached until the backward pass uses them — a real, practical constraint on how deep a network can be trained given available memory.

## Production considerations

- **Gradient checking is expensive and meant for debugging a new implementation, not routine training** — the numerical gradient in this chapter's example requires two forward passes per parameter, whereas backprop itself needs only one backward pass total; never run both in a production training loop.
- **Modern frameworks (PyTorch, TensorFlow) implement backpropagation via automatic differentiation**, removing the need to hand-derive gradients the way this chapter's example does — but understanding the mechanism directly explains why certain architectures train more easily than others.
- **A sudden `NaN` loss during training is very often a gradient problem** — an exploding gradient, or a numerically unstable operation in the forward pass — making the mechanism in this chapter the first place to look when training destabilizes.

## Common mistakes

- **Trusting that a decreasing loss means a from-scratch backprop implementation is correct**, when a subtly wrong gradient can sometimes still decrease the loss for a while before failing — gradient checking, as shown in the example, catches errors a decreasing loss alone would miss.
- **Not caching forward-pass values needed by the backward pass**, forcing a wasteful recomputation, or worse, silently using a stale or incorrect intermediate value.
- **Treating "backpropagation" and "gradient descent" as the same thing.** Backpropagation computes the gradient; gradient descent (or one of the Optimizers built on top of it, covered next) is what actually uses that gradient to update the weights.

## Interview questions

### Basic

- What does backpropagation actually compute?
- Why is backpropagation a "reverse-mode" computation, propagating from the output layer backward?

### Intermediate

- Why does reverse-mode differentiation scale better than forward-mode for a network with millions of weights and a single loss value?
- What does gradient checking verify, and why is it not used during routine training?

### Advanced

- A network's loss becomes `NaN` partway through training. How would you use what this chapter covers about backpropagation to narrow down where the problem is?
- Why does backpropagation's memory cost scale with network depth, and what trade-offs exist for training a very deep network under a fixed memory budget?
