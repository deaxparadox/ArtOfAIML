# CNNs

See [Neural Networks](neural-networks.md) for the fully-connected layer this chapter's architecture replaces for image-like data.

## What is it?

A Convolutional Neural Network (CNN) applies a small, shared set of weights — a **kernel** (or filter) — across every local region of an input, rather than connecting every input value to every unit in the next layer independently, the way the fully-connected layers in [Neural Networks](neural-networks.md) do. The same small kernel slides across the entire input, producing a **feature map**: a map of how strongly that one pattern appears at every position.

## Why does it exist?

A fully-connected layer scales terribly with image-like input: every pixel connects to every hidden unit independently, so the number of parameters grows with the *product* of input size and hidden layer width — and, critically, a fully-connected layer has to separately learn a pattern (say, a vertical edge) at every position it might appear, since nothing about the architecture assumes those positions share anything in common. CNNs exist to encode a real, useful assumption directly into the architecture instead of hoping the network learns it from data: useful local patterns (edges, textures) are largely position-independent, so the *same* small pattern-detector should be reused across every position — **weight sharing** — rather than learned separately for each one.

**Weight sharing isn't a minor optimization — it changes what the architecture can even learn efficiently.** A fully-connected layer, given enough data, could in principle learn a separate edge-detector for every position on its own — but that's an enormously wasteful use of both parameters and training data compared to a convolutional layer, which gets the "same pattern, every position" assumption for free, structurally, rather than needing to discover it. This chapter's example measures exactly how large that difference is in parameter count for one realistic case.

## How does it work?

1. **Convolution**: a small kernel (e.g. 3×3) slides across the input, computing a dot product between the kernel and the region it currently covers at every position, producing one value per position in the output feature map.
2. **Multiple kernels per layer** produce multiple feature maps, each potentially tuned to detect a different pattern — one for vertical edges, another for horizontal edges, and so on, exactly the same weight-sharing idea applied many times in parallel.
3. **Pooling** (commonly max pooling) downsamples each feature map, keeping only the strongest response within each small region — reducing the spatial resolution passed to the next layer, adding a degree of tolerance to small shifts in exactly where a pattern appeared, and reducing the computation the next layer needs to do.
4. Stacking several convolution-plus-pooling layers builds up a hierarchy: early layers detect simple local patterns (edges), and deeper layers, operating on the *feature maps* of earlier layers rather than raw pixels, can detect increasingly complex, composite patterns (shapes, parts of objects).

## Example

**Parameter count: fully-connected vs. convolutional**, for a realistic small image:

```python
img_h, img_w, channels = 64, 64, 3
hidden_units = 128

fc_params = (img_h * img_w * channels) * hidden_units + hidden_units
print("fully-connected layer params:", fc_params)

kernel_size, n_filters = 3, 16
conv_params = (kernel_size * kernel_size * channels) * n_filters + n_filters
print("conv layer params (16 filters):", conv_params)
print(f"ratio: {fc_params / conv_params:.2f}")
```

```text
fully-connected layer params: 1572992
conv layer params (16 filters): 448
ratio: 3511.14
```

A single fully-connected layer needs over 1.5 million parameters just to connect a modest 64×64 RGB image to 128 hidden units. A convolutional layer with 16 small 3×3 filters — enough to detect 16 different local patterns, applied everywhere in the image — needs only 448, over 3,500 times fewer, because the same 16 filters are reused at every position instead of learned separately for each one.

**Convolution itself, applied to a synthetic image with one clear vertical edge**:

```python
import numpy as np

image = np.array([
    [10, 10, 10, 200, 200, 200],
    [10, 10, 10, 200, 200, 200],
    [10, 10, 10, 200, 200, 200],
    [10, 10, 10, 200, 200, 200],
])

vertical_edge_kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])

def convolve2d(img, kernel):
    kh, kw = kernel.shape
    out = np.zeros((img.shape[0] - kh + 1, img.shape[1] - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(img[i:i+kh, j:j+kw] * kernel)
    return out

print(convolve2d(image, vertical_edge_kernel))
```

```text
[[   0. -570. -570.    0.]
 [   0. -570. -570.    0.]]
```

The feature map is exactly zero everywhere the image is flat (uniform 10s or uniform 200s — the kernel's weights sum to zero, so a uniform region always produces zero), and strongly non-zero (-570) exactly at the two positions straddling the actual edge between the dark and light regions. This single, small, shared kernel — applied identically at every position — correctly locates the one edge in the image without ever being told where to look.

## Where is it used?

Image classification, object detection, and any spatial or grid-structured data where local patterns matter and the same pattern can plausibly appear at different positions — the dominant architecture for computer vision tasks before, and often alongside, vision Transformers.

## Advantages

- **Dramatically fewer parameters than a fully-connected layer on image-like input**, as the example shows directly — over 3,500x fewer for one realistic case.
- **Weight sharing lets a learned pattern generalize across positions it wasn't specifically trained on**, rather than needing to be learned separately at every position.
- **Stacked layers build a natural hierarchy** from simple local patterns to complex composite ones, without that hierarchy needing to be manually engineered.

## Limitations

- **The local-pattern assumption can be wrong for some data.** Weight sharing helps precisely when patterns are position-independent — data where the same local pattern means something entirely different depending on where it appears gets less benefit from convolution's core assumption.
- **Still needs many layers and real depth to reach state-of-the-art accuracy on hard tasks**, inheriting the same training-difficulty concerns [Backpropagation](backpropagation.md) and [Optimizers](optimizers.md) cover generally, now at a scale where those concerns matter even more.
- **Pooling discards spatial precision by design** — useful for tolerance to small shifts, but a real cost for tasks needing exact spatial localization, like precise pixel-level segmentation.

## Production considerations

- **Input size changes the parameter count and compute cost non-trivially** — a larger input image means more positions for each kernel to slide across, directly increasing inference cost even though the kernel itself doesn't grow.
- **The same reproducibility and versioning discipline from [ML Workflow](../machine-learning/ml-workflow.md) applies** — a CNN's learned kernels are as much a versioned artifact as any other trained model's weights.
- **Serving a CNN at scale inherits the same latency/throughput concerns as [LLM Serving](../mlops/llm-serving.md)** — batching multiple images together is similarly far more efficient than processing them one at a time.

## Common mistakes

- **Using a fully-connected architecture on image data by default**, missing the parameter-efficiency and generalization benefits this chapter's example demonstrates directly.
- **Assuming a CNN's local-pattern assumption applies to every kind of grid-structured data**, when some data genuinely doesn't have the "same pattern, any position" property convolution is built around.
- **Over-pooling early in the network**, discarding spatial detail a later layer or task actually needed, before the network has had a chance to use it.

## Interview questions

### Basic

- What does weight sharing mean in a convolutional layer, and why does it reduce parameter count?
- What does pooling do, and what's the trade-off it introduces?

### Intermediate

- Why does a convolutional layer generalize a learned pattern across positions it wasn't directly trained on, when a fully-connected layer doesn't?
- Why does this chapter's edge-detection kernel produce exactly zero on uniform regions of the image?

### Advanced

- Design a scenario, or type of data, where the assumption underlying convolution (local patterns are position-independent) would actually hurt performance compared to a fully-connected layer.
- How would you decide how many convolutional layers and how much pooling a given image classification task actually needs, versus over- or under-provisioning the architecture?
