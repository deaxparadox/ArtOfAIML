# Plotly

See [Matplotlib](matplotlib.md#limitations) for the gap this chapter picks up directly — no built-in interactivity.

## What is it?

Plotly is a Python library for interactive, web-based visualizations — charts that support zooming, panning, and hovering for exact values, built on plotly.js and rendered in a browser rather than as a fixed image.

If Matplotlib produces a photograph of a chart, fixed the instant it's rendered, Plotly produces something closer to a small web application: the same data, but the reader can zoom into a busy region or hover over a point to see its exact value.

## Why does it exist?

[Matplotlib](matplotlib.md#limitations) named this gap directly: "static by default — no built-in interactivity." Plotly exists to fill it. Once a chart lives on a screen rather than on paper, letting the reader explore it directly — zoom into a spike, hover to check an exact value — is often more useful than the author trying to anticipate every value worth calling out in advance. It's built on plotly.js, so what it produces is a small, self-contained interactive web component, not a raster image.

**When to reach for Plotly over Matplotlib:** when a chart will actually be viewed on a screen and the audience benefits from exploring it directly — a live dashboard, a shared notebook, a report where hovering to read an exact value matters. **When Matplotlib is still the right choice:** when the output needs to be a fixed, unchanging image — a printed document, or, as this very chapter demonstrates below, a static asset committed to a repository where reproducing one exact picture is the point.

## How does it work?

- **`plotly.express`** (`px`) — a high-level, one-call interface for common chart types directly from a DataFrame, the Plotly analogue of pandas' `.plot()` sitting on top of Matplotlib.
- **`plotly.graph_objects`** (`go`) — a lower-level, explicit interface: build a `Figure`, add traces to it one at a time, the analogue of Matplotlib's object-oriented `fig, ax` interface. The example below uses this, for full control over multiple overlaid traces.
- **Output** — the same figure can render inline in a notebook, export to a standalone HTML file (`fig.write_html`) for sharing, or export to a static image (`fig.write_image`, via the separate `kaleido` package) when a fixed picture is what's actually needed.

## Example

A real limitation worth naming honestly, before writing a line of code: this handbook is static Markdown, so the actual interactive chart can't be embedded and used inline the way an image can. What follows is a static export of the figure — the honest way to show what Plotly produces inside a format that can't run JavaScript.

Reusing the exact latency and anomaly data already verified in [Standard Deviation](../statistics/standard-deviation.md#example):

```python
import statistics
import plotly.graph_objects as go

latencies = [100, 102, 98, 101, 99, 103, 97, 100, 102, 99]
mean = statistics.mean(latencies)
std = statistics.stdev(latencies)

new_value = 140
all_values = latencies + [new_value]
x = list(range(len(all_values)))
upper, lower = mean + 3 * std, mean - 3 * std

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=all_values, mode="lines+markers", name="latency (ms)"))
fig.add_trace(go.Scatter(x=x, y=[upper] * len(x), mode="lines", name="mean + 3 std", line=dict(dash="dash", color="red")))
fig.add_trace(go.Scatter(x=x, y=[lower] * len(x), mode="lines", name="mean - 3 std", line=dict(dash="dash", color="red")))
fig.add_trace(go.Scatter(x=[x[-1]], y=[new_value], mode="markers", name="anomaly (z=20.9)", marker=dict(size=14, symbol="x", color="orange")))
fig.update_layout(
    title="Request latency over time, with a 3-sigma anomaly threshold",
    xaxis_title="request index",
    yaxis_title="latency (ms)",
)

fig.write_image("assets/plotly-latency-anomaly.png")               # static export, embedded below
fig.write_html("assets/plotly-latency-anomaly.html", include_plotlyjs="cdn")  # the actual interactive version
```

![Line chart of 11 request latencies. Ten cluster tightly between roughly 97 and 103ms, within red dashed threshold lines at mean plus or minus 3 standard deviations, around 94.4 and 105.8. The eleventh point spikes to 140ms, marked with an orange X labeled "anomaly z=20.9", far above the upper threshold line.](assets/plotly-latency-anomaly.png)

This is the static export — everything Plotly can show without a browser. The [actual interactive version](assets/plotly-latency-anomaly.html) (download and open it locally) lets you hover over any point to see its exact latency, or zoom into the tightly clustered normal readings to see the small variation this static image compresses into a handful of pixels.

## Where is it used?

Live monitoring dashboards, shared analysis notebooks, and any report where a reader benefits from exploring data directly instead of relying entirely on the author's chosen static view — a direct extension of the anomaly-detection reasoning already covered in [Standard Deviation](../statistics/standard-deviation.md).

## Advantages

- **Genuine interactivity** — zoom, pan, hover for exact values — with no extra JavaScript code required from the author.
- **Scales from a quick chart to a fully custom figure**, mirroring the same two-tier design (`express` vs. `graph_objects`) Matplotlib uses (`pyplot` vs. object-oriented).
- **One figure, multiple outputs** — the same object renders inline in a notebook, exports to a standalone HTML file for sharing, or exports to a static image when that's what's actually needed.

## Limitations

- **Heavier than Matplotlib for the same chart**, since the figure carries everything needed to render and interact with it in a browser, not just draw it once.
- **Interactivity requires somewhere to actually be interactive.** Embedded in a static document — a PDF, a printed page, or, as directly demonstrated above, this handbook's own Markdown — it degrades to a static export, losing the exact feature that motivated using it.
- **A larger dependency footprint in practice.** A static export specifically needs the separate `kaleido` package, and a genuinely offline-capable HTML export needs the full plotly.js library embedded rather than loaded from a CDN.

## Production considerations

- **Dashboard payload size matters.** Embedding a Plotly chart means shipping the full interactive figure to the browser, not a lightweight static image — worth checking for a chart with a large number of points or traces, the same concern [Matplotlib](matplotlib.md#production-considerations) already raised about plotting too many raw points directly.
- **`include_plotlyjs="cdn"` vs. embedding is a real trade-off, not a default to set without thinking.** The CDN option keeps the file small (about 8.6 KB for the HTML export above) but needs internet access to load plotly.js; embedding the library directly makes the file fully offline-capable at the cost of several megabytes.
- **Static export adds a real dependency and a real failure mode.** A missing or broken `kaleido` install silently blocks every `write_image` call — a failure an interactive-only Plotly workflow wouldn't otherwise need to worry about.

## Common mistakes

- **Assuming an interactive chart will "just work" wherever it's shared**, without checking whether the destination — a static document, an email, a Markdown file — can actually run the JavaScript it depends on.
- **Choosing `include_plotlyjs="cdn"` for an export that needs to work offline**, then being confused when the chart fails to render with no internet connection.
- **Reaching for Plotly out of habit for a chart that will only ever be viewed as a fixed image**, paying the extra dependency and file-size cost of interactivity that will never actually be used.

## Interview questions

### Basic

- What does Plotly provide that Matplotlib doesn't, by default?
- Why can't a Plotly chart's interactivity be experienced in a printed document or a static image?

### Intermediate

- What's the difference between `plotly.express` and `plotly.graph_objects`, and when would you use each?
- What's the trade-off between exporting an HTML file with `include_plotlyjs="cdn"` versus embedding the library directly?

### Advanced

- You're building a production dashboard with Plotly. What would make you choose a static image export instead of the interactive chart for part of it?
- Why might a chart library built for genuine interactivity be the wrong choice for a report that will only ever be read as a PDF?
