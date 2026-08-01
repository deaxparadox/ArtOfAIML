# Observability

See [Kubernetes](kubernetes.md) for the readiness/liveness probes this chapter goes beyond.

## What is it?

Observability is the practice of instrumenting a running system so that its actual internal behavior — not just whether it's up or down — can be inspected from the outside, through logs, metrics, and traces.

[Kubernetes](kubernetes.md)'s readiness and liveness probes answer one narrow question: is this container healthy enough to receive traffic. Observability is the much richer picture behind that single yes/no answer — not just whether a service is up, but what it's actually doing, how fast, how often it fails, and why, when something does go wrong.

## Why does it exist?

Kubernetes' probes check whether a container is alive and ready — a real but narrow signal, and [Kubernetes](kubernetes.md#common-mistakes)'s own point stands: a probe checking nothing meaningful defeats its purpose. A service can pass every health check and still behave badly in ways a simple probe was never built to catch — requests taking far longer than they should, a model's predictions quietly drifting the way [ML Workflow](../machine-learning/ml-workflow.md) already warned about, or one specific type of request silently failing more often than others. Observability exists to make that deeper behavior visible and queryable after the fact, rather than something a person has to have predicted and built a specific check for in advance.

**What to instrument, and how much, is a real decision.** Logs, metrics, and traces each answer a different question and cost something different to collect and store: logging every detail of every request produces a rich record at real storage and processing cost; metrics are cheap to collect at high volume but only ever show aggregated numbers, not individual request detail. The right level of instrumentation depends on what failure modes actually matter for a given system — a fraud-detection model needs enough logged detail to investigate a specific bad decision after the fact; a low-stakes internal tool may only need basic uptime and latency metrics.

## How does it work?

**Logs** record discrete events — useful for investigating exactly what happened on one specific request after an incident. **Metrics** are numeric measurements aggregated over time — request count, error rate, latency — and, per [Mean](../statistics/mean.md) and [Standard Deviation](../statistics/standard-deviation.md)'s own point that averages hide tail behavior, are usually reported as percentiles rather than a single average. **Traces** record how a single request moved through multiple parts of a system — useful for a request that touches several services, like an agent's multi-step loop from [Agent Architecture](../agents/agent-architecture.md) or a RAG pipeline's retrieval-then-generation path from [What is RAG](../rag/what-is-rag.md) — showing where time was actually spent, not just that the whole thing was slow.

## Example

Instrumenting the house-price [FastAPI](fastapi.md) service with real request logging, then applying [Standard Deviation](../statistics/standard-deviation.md)'s own z-score anomaly detection to the logged latencies — and hitting exactly the pitfall that chapter warned about:

```python
import time
from fastapi import FastAPI
from pydantic import BaseModel

logs = []
app = FastAPI()

class HouseFeatures(BaseModel):
    size_sqm: float
    simulate_slow_downstream: bool = False

@app.post("/predict")
def predict(features: HouseFeatures):
    start = time.perf_counter()
    if features.simulate_slow_downstream:
        time.sleep(0.05)  # simulate a slow downstream call
    prediction = model.predict([[features.size_sqm]])[0]
    logs.append({"duration_ms": (time.perf_counter() - start) * 1000})
    return {"predicted_price_thousands": float(prediction)}
```

Running 7 ordinary requests plus one deliberately slow one produced these logged durations (ms): `[0.42, 0.23, 0.22, 0.20, 0.22, 0.21, 0.21, 50.43]` — the slow request is roughly 200x the others. Computing a z-score the naive way, with the anomaly included in its own baseline:

```text
mean=6.52ms, std=17.74ms, z=2.47   <- below a typical 3-sigma alert threshold
```

The obviously anomalous request isn't flagged — its own extreme value inflated the mean and standard deviation it was being measured against. Computing the baseline from only the 7 ordinary requests instead:

```text
baseline mean=0.244ms, std=0.078ms, z=642.81   <- unmistakably flagged
```

This is exactly the pitfall [Standard Deviation](../statistics/standard-deviation.md#limitations) already named: "if the window used to compute mean and standard deviation already contains anomalies... the baseline itself is wrong." A monitoring baseline needs to be computed from a clean reference window, not from whatever data happens to include the very anomaly it's meant to catch.

## Where is it used?

Any production service, but especially ones already covered in this section: a [FastAPI](fastapi.md)-served model, a [Kubernetes](kubernetes.md) deployment, or the multi-step pipelines from [Agent Architecture](../agents/agent-architecture.md) and [What is RAG](../rag/what-is-rag.md) where a slowdown could be hiding in any one of several steps.

## Advantages

- **Makes a system's actual behavior inspectable after the fact**, not just its up/down status.
- **Metrics and logs together let a specific incident be diagnosed precisely**, rather than guessed at from a single "something is wrong" alert.
- **Percentile-based metrics catch tail problems** that an average, per [Mean](../statistics/mean.md) and [Standard Deviation](../statistics/standard-deviation.md), would hide entirely.

## Limitations

- **A naive anomaly-detection baseline can be contaminated by the very anomaly it's meant to catch**, as the example shows directly — the detection method needs a clean reference window, which isn't automatic.
- **More instrumentation means more data to store and process** — a real, ongoing cost that scales with traffic, not a one-time setup cost.
- **Observability data shows what happened, not automatically why.** A latency spike in a trace or metric still needs a person, or another system, to investigate and explain it.

## Production considerations

- **The reference window used for any anomaly threshold needs to be chosen deliberately**, kept clean of known anomalies, and refreshed as normal behavior legitimately changes over time.
- **Percentiles (p50, p95, p99) need to be tracked alongside averages, not instead of them** — each answers a different question about the same underlying data.
- **Traces matter most exactly where a request touches multiple steps or services.** A multi-agent system or RAG pipeline benefits from tracing far more than a single, simple endpoint does, since there are more places for time to disappear.

## Common mistakes

- **Computing an anomaly-detection baseline over a window that includes the anomaly itself**, silently raising the threshold enough to hide it — exactly what this chapter's example demonstrates happening.
- **Instrumenting only averages**, missing the tail behavior that actually matters for user experience — the same blind spot [Mean](../statistics/mean.md) already raised.
- **Treating "we have logs" as equivalent to "we have observability,"** without metrics or traces to answer the questions logs alone can't.

## Interview questions

### Basic

- What's the difference between logs, metrics, and traces?
- Why isn't a passing Kubernetes health check sufficient observability on its own?

### Intermediate

- Why did the anomalous request in this chapter's example fail to trigger a 3-sigma alert on the first attempt?
- Why are latency metrics usually reported as percentiles instead of a single average?

### Advanced

- Design a monitoring approach for a multi-step agent or RAG pipeline where a slowdown could originate in any one of several steps. What would you instrument, and why?
- How would you build an anomaly-detection baseline that stays valid over time, given that "normal" behavior can legitimately shift?
