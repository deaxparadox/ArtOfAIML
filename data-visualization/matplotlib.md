# Matplotlib

## What is it?

Matplotlib is the foundational Python plotting library — the base that many other Python visualization tools (pandas' `.plot()`, seaborn) are built on top of. It gives explicit, low-level control over a plot: figures, axes, lines, colors, and labels, built up step by step.

Think of Matplotlib as a drafting table, not a stencil kit. You place every element deliberately — this line, this color, this axis label — rather than calling one function that hands back a fully-styled chart. Higher-level tools trade away some of that control in exchange for speed.

## Why does it exist?

[ML Workflow](../machine-learning/ml-workflow.md#how-does-it-work) named EDA as a distinct step precisely because numbers alone often don't reveal what's actually wrong with data. The same-mean-different-variance datasets from [Variance](../statistics/variance.md) or the skewed salary example from [Mean](../statistics/mean.md) are both far faster to spot in a plot than in a printed statistic. Matplotlib exists to bring that kind of visualization natively into Python — modeled explicitly on MATLAB's plotting interface, so scientific Python code could compute and visualize without switching tools or languages.

**A real decision: which interface to use.** `plt.plot(...)` — the "pyplot" interface — is a MATLAB-style shortcut that implicitly tracks a "current" figure and axes behind the scenes. It's fine for one quick, throwaway plot. The moment a figure needs multiple panels, or gets built inside a function and reused, the explicit object-oriented interface (`fig, ax = plt.subplots()`, then calling methods directly on `ax`) is the safer choice — implicit global state gets confusing fast once more than one plot is involved, which is exactly why the example below uses it.

## How does it work?

- **Figure** — the whole canvas.
- **Axes** — one plot area within a figure; a single figure can hold several.
- **pyplot interface** (`plt.plot`, `plt.show`) — implicitly manages a "current" figure and axes for you.
- **Object-oriented interface** (`fig, ax = plt.subplots()`) — makes the figure and each axes explicit objects you call methods on directly, scaling cleanly to multiple subplots.

## Example

Turning the polynomial-fitting story from [Bias vs Variance](../machine-learning/bias-vs-variance.md#example) into an actual picture, using the object-oriented interface for its three side-by-side panels:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
X = np.linspace(0, 10, 30)
y = np.sin(X) + rng.normal(0, 0.2, size=X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

x_smooth = np.linspace(0, 10, 300)
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

for ax, degree in zip(axes, [1, 9, 15]):
    coeffs = np.polyfit(X_train, y_train, degree)
    y_smooth = np.polyval(coeffs, x_smooth)

    ax.scatter(X_train, y_train, label="train", s=25)
    ax.scatter(X_test, y_test, label="test", s=25)
    ax.plot(x_smooth, y_smooth, color="black", linewidth=1.5)
    ax.set_ylim(-2, 2)
    ax.set_title(f"degree {degree}")

axes[0].legend(loc="upper right", fontsize=8)
fig.savefig("bias-variance-fit.png", dpi=120)
```

![Three side-by-side plots showing a degree-1 polynomial as a nearly flat line missing the sine curve entirely, a degree-9 polynomial tracking the sine curve closely, and a degree-15 polynomial tracking the curve well through most of the range but swinging wildly near x=8-9, close to the edge of the training data.](assets/matplotlib-bias-variance-fit.png)

The picture shows exactly what the numbers in [Bias vs Variance](../machine-learning/bias-vs-variance.md#example) already established: degree 1 is a nearly flat line that misses the sine shape completely (high bias), degree 9 tracks the curve closely (a good fit), and degree 15 tracks the curve well through most of the range but swings wildly near the edge of the data, around x = 8–9 — visibly unstable exactly where that chapter's verified test error exploded to 1892.

## Where is it used?

The EDA step of [ML Workflow](../machine-learning/ml-workflow.md), reporting spread and distribution shape visually (the same ideas covered numerically in [Mean](../statistics/mean.md), [Variance](../statistics/variance.md), and [Standard Deviation](../statistics/standard-deviation.md)), model diagnostics like the fit comparison above, and as the actual rendering engine several higher-level libraries call underneath their own simpler interface.

## Advantages

- **Fine-grained control over every element of a plot** — exact colors, labels, layout, and annotations, useful for publication-quality or precisely specified output.
- **The de facto standard** — broadly compatible, extensively documented, and the actual rendering engine behind several higher-level plotting libraries.
- **Scales from a one-line throwaway plot to a carefully composed multi-panel figure**, depending on which interface is used.

## Limitations

- **Verbose for standard chart types.** A single boxplot-by-category takes noticeably more raw Matplotlib code to reproduce than the equivalent one-line call in a higher-level library built on top of it.
- **Static by default.** No built-in interactivity — zooming, hovering for a tooltip — the way the next chapter's library, Plotly, provides natively.
- **The pyplot interface's implicit "current axes" state is easy to misuse.** Once a script has more than one plot, it's easy to have a call land on the wrong axes without an obvious error telling you so.

## Production considerations

- **Plotting millions of raw points directly is slow and unreadable.** Production dashboards usually pre-aggregate first — a 2D histogram, a sampled subset — rather than scattering raw data at that scale.
- **Automated pipelines need a non-interactive backend.** Generating a chart inside a scheduled job or server process (with no display attached) requires a backend built for that, rather than one that assumes an interactive window.
- **Chart-generation code needs the same reproducibility discipline as a model.** The exact code, library version, and input data behind a report or monitoring chart should be recoverable later — the same concern [ML Workflow](../machine-learning/ml-workflow.md#production-considerations) already raised about model pipelines generally.

## Common mistakes

- **Mixing the pyplot and object-oriented interfaces inconsistently** across a script with multiple plots, then being confused when a call lands on the wrong figure or axes.
- **Plotting a very large number of raw points directly**, producing a chart too dense to actually read, instead of aggregating first.
- **Forgetting to save or display the figure** and assuming a plot was produced when nothing was ever rendered or written to disk.

## Interview questions

### Basic

- What's the difference between a Figure and an Axes in Matplotlib?
- Why would you choose a plot over a printed summary statistic during EDA?

### Intermediate

- What's the difference between Matplotlib's pyplot interface and its object-oriented interface, and when would you prefer one over the other?
- Why can plotting millions of raw data points directly produce a chart that's harder to read than one built from aggregated data?

### Advanced

- Why does generating a plot inside an automated, non-interactive pipeline require a different Matplotlib backend than an interactive session?
- The degree-15 fit in this chapter's example swings wildly near the edge of the training data. How does that same instability show up in the numbers from [Bias vs Variance](../machine-learning/bias-vs-variance.md), and why do both appear together?
