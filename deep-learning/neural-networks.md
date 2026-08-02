# Neural Networks

See [Supervised Learning](../machine-learning/supervised-learning.md) for the single-layer linear model (Logistic Regression) this chapter builds directly on top of.

## What is it?

A neural network is a stack of layers of simple computational units — each computing a weighted sum of its inputs, adding a bias, then passing the result through a non-linear **activation function** — where each layer's output becomes the next layer's input. Stacking multiple such layers lets the network represent functions no single layer structurally can, which is exactly where "deep" in deep learning comes from.

[Supervised Learning](../machine-learning/supervised-learning.md#how-does-it-work) already covered Logistic Regression: a linear combination of inputs, squashed through a sigmoid into a probability. That's not a simplified neural network — it *is* one, specifically a network with no hidden layers at all: one layer, one unit, one activation. A neural network in the usual sense adds one or more **hidden layers** in between the input and the output, each applying its own weighted sum and activation, before the final layer produces a prediction.

## Why does it exist?

A single linear layer, no matter how it's trained, can only ever represent a linear decision boundary (or, with a sigmoid, a linear boundary in probability space) — [Supervised Learning](../machine-learning/supervised-learning.md#example) already showed this failing concretely, losing to a tree or KNN on curved data. Neural networks exist to remove that ceiling entirely: stacking layers with a non-linear activation between them lets the network compose simple linear pieces into an arbitrarily complex, non-linear function — not just a curved boundary, but, with enough width and depth, any continuous function to arbitrary precision (the **universal approximation** result).

**Single-layer vs. multi-layer is a real, structural line, not a matter of degree.** A single-layer network — Logistic Regression — is exactly as representationally limited as any other linear model, regardless of how it's trained or regularized: it cannot represent XOR, the classic example of a function with no linear boundary that separates it, no matter how the two input features are weighted. Adding even one hidden layer with a non-linear activation crosses that line: the hidden layer can learn an intermediate representation where the originally inseparable problem becomes separable to the final layer. This is the entire reason "hidden layers," not just "more features" or "more regularization," are the lever that matters here.

## How does it work?

1. **Forward pass**: each layer computes `activation(X @ W + b)` — a weighted sum of the previous layer's output, plus a bias, passed through a non-linear activation function (its own chapter, [Activation Functions](activation-functions.md)).
2. The output of one layer becomes the input to the next, so each additional hidden layer builds a new representation on top of the previous one's output, not directly on the raw input.
3. The final layer's output is the network's prediction — a class probability (classification) or a continuous value (regression), the same output types [Supervised Learning](../machine-learning/supervised-learning.md#how-does-it-work) already established.
4. Training adjusts every layer's weights and biases via **gradient descent** — the same mechanism [Supervised Learning](../machine-learning/supervised-learning.md#example) already verified for Linear/Logistic Regression — computing gradients through every layer via **backpropagation** (its own chapter, Backpropagation).

## Example

**A manual forward pass** through a tiny 2-input, 2-hidden-unit, 1-output network, showing the mechanism directly:

```python
import numpy as np

rng = np.random.default_rng(0)
x = np.array([1.0, 0.0])
W1, b1 = rng.normal(size=(2, 2)), rng.normal(size=2)
W2, b2 = rng.normal(size=(2, 1)), rng.normal(size=1)

relu = lambda z: np.maximum(0, z)
sigmoid = lambda z: 1 / (1 + np.exp(-z))

z1 = x @ W1 + b1
a1 = relu(z1)
z2 = a1 @ W2 + b2
a2 = sigmoid(z2)
print("hidden pre-activation:", z1)
print("hidden post-ReLU:", a1)
print("final output:", a2)
```

```text
hidden pre-activation: [-0.410  0.229]
hidden post-ReLU:      [ 0.000  0.229]
final output:          [0.381]
```

ReLU zeroes out the first hidden unit's negative pre-activation entirely, while passing the second one through unchanged — exactly the mechanism, applied once per unit, per layer, that lets a network selectively "turn off" parts of its own computation depending on the input.

**Single-layer vs. multi-layer on XOR** — the textbook case with no linear solution at all:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # XOR: label is 1 exactly when inputs differ

logreg = LogisticRegression().fit(X, y)
print("LogisticRegression:", logreg.predict(X), accuracy_score(y, logreg.predict(X)))

mlp = MLPClassifier(hidden_layer_sizes=(4,), activation="relu", max_iter=5000, random_state=0).fit(X, y)
print("MLPClassifier (1 hidden layer):", mlp.predict(X), accuracy_score(y, mlp.predict(X)))
```

```text
LogisticRegression:            [0 0 0 0]  accuracy: 0.5
MLPClassifier (1 hidden layer): [0 1 1 0]  accuracy: 1.0
```

`LogisticRegression` — a single linear layer — can't do better than predicting the majority class; XOR has no linear boundary to find, so training simply fails to separate the classes at all. Adding one hidden layer with only 4 units solves it perfectly. Nothing about the training procedure changed — same data, same kind of gradient-based fitting — only the architecture gained the one thing a single layer structurally lacks: an intermediate, non-linear representation.

## Where is it used?

Every architecture covered elsewhere in this handbook is built from this same foundation: [Transformers](../llms/transformers.md) and [Attention](../llms/attention.md) stack many such layers with a specific connectivity pattern, and CNNs and RNNs / LSTMs (their own upcoming chapters) are different structural variations on the same weighted-sum-plus-activation building block.

## Advantages

- **Removes the linear-boundary ceiling entirely**, as the XOR example shows directly — a problem with literally no linear solution becomes solvable with one added layer.
- **Universal approximation**: with enough width and depth, a network can represent any continuous function to arbitrary precision, not just the specific non-linear shapes a tree or kernel method happens to capture well.
- **Composable, reusable representation**: each hidden layer builds on the previous layer's learned representation, letting depth do work that hand-engineering more features would otherwise have to do manually.

## Limitations

- **More parameters means more data (and compute) is needed to fit them well** — a network with many layers and units can overfit a small dataset far more easily than the single-layer models covered in [Supervised Learning](../machine-learning/supervised-learning.md).
- **Loses the direct interpretability of a linear model.** A `LogisticRegression` coefficient has a direct meaning; a weight three layers deep in a network generally doesn't, on its own.
- **Training is a genuinely harder optimization problem** than fitting a single linear layer — see [Backpropagation](backpropagation.md) and [Optimizers](optimizers.md) for what that actually requires in practice.

## Production considerations

- **Architecture (depth, width) is a real, tunable hyperparameter**, not a fixed given — too small underfits the way a single linear layer does on XOR; too large risks the overfitting [Bias vs Variance](../machine-learning/bias-vs-variance.md) already covered, now with a much larger parameter space to overfit with.
- **Every layer added increases inference latency and memory cost**, directly relevant to the serving concerns already covered in [LLM Serving](../mlops/llm-serving.md), just at a smaller scale here.
- **Reproducibility needs the same discipline as any model** in [ML Workflow](../machine-learning/ml-workflow.md) — weight initialization, training order, and even hardware-level non-determinism can all affect the exact result of training the same architecture twice.

## Common mistakes

- **Reaching for a deep network on a problem a linear model already solves well**, adding real training cost, overfitting risk, and lost interpretability for no corresponding gain.
- **Assuming more layers always helps**, without checking whether the added depth is actually needed to represent the target function, versus just adding more ways to overfit.
- **Treating architecture as separate from training difficulty.** A deeper network isn't just "the same training, more layers" — see [Backpropagation](backpropagation.md) and [Optimizers](optimizers.md) for why depth itself makes training harder.

## Interview questions

### Basic

- What is a hidden layer, and what does it add that a single linear layer doesn't have?
- Why can't Logistic Regression solve XOR, no matter how it's trained?

### Intermediate

- What does "universal approximation" mean, and why does it require a non-linear activation function specifically?
- Why does adding hidden layers increase both a network's representational power and its overfitting risk at the same time?

### Advanced

- You have a tabular dataset where a Random Forest already performs well. What would make you consider a neural network instead, given the added training and interpretability cost this chapter covers?
- The XOR example needed only 4 hidden units to solve a problem with 4 data points. What does that suggest about how much representational capacity is actually needed for a given problem, versus how much a real, larger dataset with a subtler pattern might need?
