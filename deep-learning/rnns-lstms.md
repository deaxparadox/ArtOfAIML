# RNNs / LSTMs

See [Activation Functions](activation-functions.md) for the vanishing gradient problem this chapter shows recurring across time steps instead of layers, and [Attention](../llms/attention.md) for the architecture that displaced RNNs for most large-scale sequence modeling.

## What is it?

A Recurrent Neural Network (RNN) processes a sequence one element at a time, maintaining a **hidden state** that's updated at every step and carries information forward from every previous element — unlike the architectures in [Neural Networks](neural-networks.md) and [CNNs](cnns.md), which process their entire input at once with no inherent notion of "what came before." An LSTM (Long Short-Term Memory) is a specific, more sophisticated RNN variant, adding a separate **cell state** and learned gates that control what information is kept, added, or exposed at each step — built specifically to fix a real, severe problem with plain RNNs.

## Why does it exist?

Real sequences — text, time series, audio — have variable length and inherent order, where what came earlier genuinely changes how to interpret what comes later. A plain RNN handles this by reusing the *same* weight matrix at every time step, applying it to both the current input and the previous hidden state — but that repetition is exactly the problem: backpropagation through time multiplies the gradient by that same weight matrix (and an activation derivative) once per time step, precisely the mechanism [Activation Functions](activation-functions.md#why-does-it-exist) already showed causing vanishing gradients across layers — except now the "layers" are time steps, and a long sequence can have far more of them than a deep network has layers.

LSTMs exist specifically to fix this. Instead of forcing information to survive a repeated matrix multiplication at every step, an LSTM maintains a separate cell state updated **additively** — `new_cell_state = forget_gate * old_cell_state + input_gate * candidate` — controlled by gates that are themselves learned. Addition, unlike repeated multiplication by the same matrix, doesn't shrink a gradient by a fixed factor at every single step, giving gradients a much more direct path back across time.

**RNN/LSTM vs. Transformer — the real reason one displaced the other for most large-scale sequence tasks.** An RNN's hidden state at step `t` depends on step `t-1`'s, which depends on `t-2`'s, and so on — inherently sequential, with no way to compute step 100 before step 99 is done. [Attention](../llms/attention.md) connects any two positions in a sequence in a single step, computable for the entire sequence in parallel, exactly the property that made Transformers dramatically more efficient to train at scale. RNNs and LSTMs remain genuinely useful specifically where that sequential nature is an advantage rather than a cost: low-latency streaming inference (processing one new data point as it arrives, without needing the whole sequence upfront) and simpler tasks where a Transformer's much larger data and compute requirements aren't justified.

## How does it work?

- **Plain RNN**: at each time step `t`, `hidden_state[t] = tanh(Wx @ x[t] + Wh @ hidden_state[t-1] + b)` — the same `Wx` and `Wh` reused at every step.
- **Backpropagation through time** computes the gradient by unrolling the recurrence and applying the chain rule backward across time steps, exactly as [Backpropagation](backpropagation.md) does across layers — the same weight matrix and activation derivative get multiplied in at every single step going backward.
- **LSTM gates**: a **forget gate** decides what fraction of the previous cell state to keep; an **input gate** decides what new candidate information to add; the cell state updates additively from those two; an **output gate** decides what part of the (now updated) cell state gets exposed as the new hidden state.

## Example

**Vanishing gradients over time, measured directly** — a plain RNN's gradient reaching the first time step after a 30-step sequence, versus an LSTM cell state's gradient over the same span:

```python
import numpy as np

rng = np.random.default_rng(0)
hidden_size, n_steps = 8, 30
tanh, tanh_grad = np.tanh, lambda h: 1 - h**2
Wh = rng.normal(size=(hidden_size, hidden_size), scale=0.5)
Wx = rng.normal(size=(1, hidden_size), scale=0.5)

# plain RNN forward pass
h = np.zeros((1, hidden_size))
hs = [h]
xs = rng.normal(size=(n_steps, 1))
for t in range(n_steps):
    h = tanh(xs[t:t+1] @ Wx + h @ Wh)
    hs.append(h)

# backprop through time
grad = np.ones((1, hidden_size))
for t in reversed(range(n_steps)):
    grad = (grad * tanh_grad(hs[t+1])) @ Wh.T
print("plain RNN gradient reaching step 0:", np.linalg.norm(grad))

# LSTM cell-state gradient: additive update, scaled only by the forget gate at each step
z = rng.normal(size=(n_steps, hidden_size), scale=0.3)
forget_gates = 1 / (1 + np.exp(-(z + 2.0)))  # biased toward "remember", a standard LSTM init trick
print("mean forget gate value:", forget_gates.mean())

cell_grad = np.ones(hidden_size)
for t in reversed(range(n_steps)):
    cell_grad = cell_grad * forget_gates[t]
print("LSTM cell-state gradient reaching step 0:", np.linalg.norm(cell_grad))
```

```text
plain RNN gradient reaching step 0: 0.00315
mean forget gate value: 0.874
LSTM cell-state gradient reaching step 0: 0.05010
```

After 30 time steps, the plain RNN's gradient has shrunk to 0.00315 — repeated multiplication by the same weight matrix and `tanh`'s derivative, one factor per step, compounding the same way [Activation Functions](activation-functions.md#example) showed across network depth. The LSTM's cell-state gradient, propagated through an additive update scaled only by a forget gate biased toward "remember" (mean 0.874), reaches 0.0501 over the identical span — nearly 16x larger. Both still shrink somewhat over 30 steps, but the additive path degrades far more gracefully than the plain RNN's repeated matrix multiplication.

## Where is it used?

Time-series forecasting, on-device or streaming inference where processing one input at a time with low latency matters more than a Transformer's full-sequence parallelism, and sequence tasks with limited training data where a much smaller RNN/LSTM is a better fit than a data-hungry Transformer.

## Advantages

- **Naturally handles variable-length sequences** with a fixed-size hidden state, regardless of how long the sequence is.
- **LSTM's additive cell-state update measurably preserves gradient magnitude better than a plain RNN**, as the example shows directly — nearly 16x more gradient reaching the first time step over the same 30-step span.
- **Processes one element at a time**, making it a natural fit for genuinely streaming, low-latency inference where a Transformer's need for the whole sequence upfront is a real mismatch.

## Limitations

- **Inherently sequential — step `t` depends on step `t-1`'s completed computation**, blocking the kind of full-sequence parallelism [Attention](../llms/attention.md) provides, and directly limiting how efficiently an RNN/LSTM can be trained at scale.
- **Even an LSTM's improved gradient flow still degrades over very long sequences**, as the example's own numbers show — better than a plain RNN, not immune to the underlying problem.
- **Generally requires more careful tuning (gate initialization, sequence length) than a well-established Transformer recipe** for a comparably difficult task, at least at the scale most current sequence-modeling work operates.

## Production considerations

- **Sequence length directly drives both training cost and gradient-flow quality** — per this chapter's own measured degradation over 30 steps, a much longer sequence should be expected to degrade further, worth testing directly rather than assuming.
- **Streaming/low-latency use cases are the strongest real reason to choose an RNN/LSTM over a Transformer today** — if the actual requirement is processing one new input as it arrives, that architectural fit can outweigh a Transformer's raw accuracy advantage.
- **The forget-gate bias initialization used in this chapter's example (biasing toward "remember") is a standard, real practical trick**, not an arbitrary choice — initializing gates neutrally can leave an LSTM with much of a plain RNN's vanishing-gradient problem in early training.

## Common mistakes

- **Assuming an LSTM is immune to vanishing gradients entirely**, when this chapter's own numbers show its gradient still shrinks over a long sequence — just far less severely than a plain RNN's.
- **Defaulting to an RNN/LSTM for a large-scale sequence task**, without considering whether a Transformer's parallelism and long-range connectivity from [Attention](../llms/attention.md) would train faster and more effectively at the available scale.
- **Neglecting gate initialization**, leaving an LSTM to rediscover the "remember by default" behavior from scratch during training rather than starting with it.

## Interview questions

### Basic

- What does a hidden state carry forward in an RNN?
- What's the structural difference between a plain RNN's update and an LSTM's cell-state update?

### Intermediate

- Why does a plain RNN suffer from vanishing gradients over long sequences, using the same mechanism that causes it across network depth?
- Why does an LSTM's additive cell-state update preserve gradient magnitude better than a plain RNN's repeated matrix multiplication?

### Advanced

- This chapter's example showed the LSTM's gradient advantage shrinking the forget gate's "remember" bias would reduce. How would you expect the 16x gradient-preservation advantage to change with a different, less favorable gate initialization?
- Given that Transformers dominate large-scale sequence modeling, construct a concrete scenario where choosing an RNN/LSTM over a Transformer would still be the right engineering decision.
