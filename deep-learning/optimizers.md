# Optimizers

See [Backpropagation](backpropagation.md) for the gradient this chapter's algorithms actually use, and [Supervised Learning](../machine-learning/supervised-learning.md#example) for the plain gradient descent update this chapter builds on top of.

## What is it?

An optimizer is the algorithm that turns a computed gradient into an actual weight update. [Supervised Learning](../machine-learning/supervised-learning.md#example) verified plain gradient descent directly: subtract the gradient, scaled by a fixed learning rate, from the weights. An optimizer is anything more sophisticated than that single rule — incorporating information from past gradients (**momentum**) or adapting the effective step size separately for each parameter (**Adam**) — built specifically to converge faster and more reliably than the plain update alone.

## Why does it exist?

Plain gradient descent's single, fixed-size step works fine on a well-behaved loss surface, but a real network's loss surface is rarely that cooperative — some directions have much steeper curvature than others (an **ill-conditioned** surface), and a fixed learning rate that's appropriately sized for the steep direction is far too small for the shallow one, leaving convergence painfully slow along it. Optimizers exist to handle this mismatch: by accumulating information about past gradients, they can move faster in directions that are consistently improving and adapt how aggressively to move in each direction independently, rather than treating every parameter identically.

**Plain SGD vs. Momentum vs. Adam — a real, measurable trade-off, not just "newer is better."** Plain SGD is simple and well-understood but genuinely struggles on an ill-conditioned surface, as this chapter's example shows directly. Momentum accumulates a running average of past gradients, accelerating consistent-direction movement and damping oscillation — but, as the example also shows, it can itself overshoot before settling, a real cost of building up speed. Adam goes further, tracking both a momentum-like average and a per-parameter estimate of gradient magnitude, adapting the effective step size separately for every parameter — converging fast with comparatively little tuning, though it doesn't always generalize as well as a carefully hand-tuned SGD-with-momentum setup in every case, a genuine, documented trade-off rather than Adam being strictly superior.

## How does it work?

- **Plain (S)GD**: `w = w - lr * grad`. Every parameter uses the exact same step size, in the exact direction the current gradient points.
- **Momentum**: maintains a running exponential average of the gradient, `v = beta * v + (1 - beta) * grad`, and updates using `v` instead of the raw gradient. A consistent gradient direction builds up speed over successive steps; an inconsistent one partially cancels out.
- **Adam**: maintains both a momentum-like first-moment average (`m`, the mean of recent gradients) and a second-moment average (`v`, the mean of recent *squared* gradients), then updates using `m / (sqrt(v) + eps)` — dividing by each parameter's own recent gradient magnitude. A parameter with a history of large gradients gets a smaller effective step; one with a history of small gradients gets a larger effective step, independently of every other parameter.

## Example

**The classic ill-conditioned bowl**: a loss surface 50x steeper along one axis than the other — `loss(w) = 0.5 * (50*w_x² + 1*w_y²)` — the exact situation where a single fixed learning rate serves one direction well and the other poorly.

```python
import numpy as np

A = np.array([50.0, 1.0])
grad = lambda w: A * w
loss = lambda w: 0.5 * np.sum(A * w**2)

def sgd(lr=0.018, steps=100):
    w = np.array([1.0, 1.0])
    for _ in range(steps):
        w = w - lr * grad(w)
    return w

def momentum(lr=0.018, beta=0.9, steps=100):
    w, v = np.array([1.0, 1.0]), np.zeros(2)
    for _ in range(steps):
        v = beta * v + (1 - beta) * grad(w)
        w = w - lr * v
    return w

def adam(lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, steps=100):
    w, m, v = np.array([1.0, 1.0]), np.zeros(2), np.zeros(2)
    for t in range(1, steps + 1):
        g = grad(w)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat, v_hat = m / (1 - beta1**t), v / (1 - beta2**t)
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
    return w

for name, fn in [("plain SGD", sgd), ("SGD + momentum", momentum), ("Adam", adam)]:
    w = fn()
    print(f"{name}: w={w}, final loss={loss(w):.6f}")
```

```text
plain SGD:      w=[0.       0.163], final loss=0.013221
SGD + momentum: w=[0.00292  0.137],  final loss=0.009594
Adam:           w=[0.00294  0.00294], final loss=0.000220
```

Plain SGD converges its steep direction (`w_x`) almost immediately but crawls in the shallow direction (`w_y`, still at 0.163 after 100 steps) — the same learning rate is simultaneously "correctly sized" for one axis and far too small for the other. Momentum does slightly better overall by the end, but along the way it genuinely overshoots: `w_x` swings to **-0.587** by step 10 before oscillating back toward zero — real, measured overshoot from the accumulated velocity, not a hypothetical risk. Adam converges both directions to almost identical magnitudes (0.00294 each, despite the 50x difference in curvature) and reaches by far the lowest final loss — exactly because it adapts each parameter's effective step size to its own gradient history, independently of the other.

## Where is it used?

Training every neural network in this handbook and beyond — Adam (or a close variant, AdamW) is the default optimizer for most modern deep learning, including [Transformers](../llms/transformers.md)-based LLM pretraining and fine-tuning, per [Fine-Tuning](../llms/fine-tuning.md).

## Advantages

- **Handles ill-conditioned loss surfaces far better than plain gradient descent**, as the example shows directly — Adam reaches a loss over 50x lower than plain SGD after the same number of steps.
- **Momentum smooths out inconsistent gradient directions**, accelerating convergence along directions that consistently improve.
- **Adam requires comparatively little learning-rate tuning** to get reasonable results across very different parameters, since it adapts per-parameter automatically.

## Limitations

- **Momentum can overshoot**, as the example's own oscillation shows — building up speed helps on average but isn't free of its own instability.
- **Adam's adaptive per-parameter steps don't always generalize as well as well-tuned SGD with momentum**, a real, documented trade-off in some architectures and tasks, not a universal ranking of "Adam is simply better."
- **Every optimizer still depends on a correctly-computed gradient from [Backpropagation](backpropagation.md)** — no optimizer can fix a genuinely wrong or vanished gradient, only work with whatever gradient it's given.

## Production considerations

- **Optimizer choice and learning rate are both real hyperparameters that need tuning together**, not independently — the same learning rate can be reasonable for one optimizer and unstable for another, as this chapter's example's own parameter choices show.
- **Adam's memory cost is roughly double plain SGD's**, since it stores both a first- and second-moment estimate per parameter — a real, if usually acceptable, cost at large model scale.
- **A training run that suddenly diverges (loss spikes to `NaN`) is sometimes an optimizer/learning-rate mismatch**, not necessarily the vanishing/exploding gradient issue [Activation Functions](activation-functions.md) covered — both are worth checking.

## Common mistakes

- **Using the same learning rate across every optimizer without re-tuning**, then concluding one optimizer is "worse" when the actual problem is a learning rate suited to a different update rule.
- **Assuming Adam is always the right choice**, when this chapter's own limitations section names a real, documented case where it doesn't generalize as well as a carefully tuned alternative.
- **Not recognizing momentum-driven oscillation as expected behavior**, and reflexively lowering the learning rate rather than understanding it as a real, sometimes-acceptable cost of building up speed.

## Interview questions

### Basic

- What does an optimizer do that plain gradient descent doesn't?
- What's the difference between momentum and Adam, at a high level?

### Intermediate

- Why does an ill-conditioned loss surface cause plain gradient descent to converge unevenly across directions, as this chapter's example shows?
- Why can momentum overshoot, and why does it still often converge faster overall despite that?

### Advanced

- This chapter's example showed Adam converging both directions of a 50x ill-conditioned surface to nearly identical magnitudes. Why does normalizing by each parameter's own gradient history achieve that, and what would happen if the surface were ill-conditioned in a way that changed over the course of training?
- Design an experiment to determine whether Adam's faster convergence on a specific task is coming at the cost of worse generalization, versus a well-tuned SGD-with-momentum baseline.
