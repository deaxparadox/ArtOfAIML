# Polars

See [Pandas](pandas.md#limitations) for the two gaps — memory ceiling and single-threaded execution — this chapter picks up directly.

## What is it?

Polars is a DataFrame library for Python, implemented in Rust on top of Apache Arrow's columnar memory format. It offers the same DataFrame concept as pandas, but with two execution modes: **eager** (run each operation immediately, like pandas) and **lazy** (build a full query plan first, optimize it, then execute it in one pass).

Pandas runs each line of code the moment you write it — closer to typing individual commands into a shell. Polars' lazy mode is closer to writing one SQL query and letting a query planner figure out the most efficient way to execute the whole thing at once, including skipping data it can prove it will never need.

## Why does it exist?

[Pandas](pandas.md#limitations) named the two gaps this fills directly: it isn't built for out-of-memory data, and row-wise Python operations (`.apply()`) fall back to slow, effectively single-threaded execution. Polars addresses both: it's implemented in Rust with genuine multi-threaded execution by default, and its lazy API builds a complete query plan before running anything — letting it read only the columns actually needed and filter rows as early as possible, instead of executing each pandas-style line eagerly and in isolation.

**When to reach for Polars over pandas:** when data volume is large enough that pandas' single-threaded, eager, full-memory-load model becomes the actual bottleneck, or specifically when a multi-step pipeline would benefit from a query planner reasoning about the whole thing before executing it. **When pandas is still the better choice:** smaller datasets, or an existing pandas-based codebase — a large share of the Python data ecosystem (plotting libraries, some ML tooling) still expects a pandas DataFrame specifically, not a Polars one.

## How does it work?

- The same labeled, columnar DataFrame concept as pandas, backed by Arrow's memory format instead of raw NumPy arrays underneath.
- **Eager mode** (`pl.DataFrame`) behaves like pandas — every operation runs immediately.
- **Lazy mode** (`.lazy()`, or `pl.scan_csv` for reading a file lazily) builds a query plan you materialize explicitly with `.collect()`. The planner can push filters and column selection down to the earliest possible point in the plan, and parallelize independent parts of it across CPU cores automatically.

## Example

The same customer table from [Pandas](pandas.md), in Polars:

```python
import polars as pl

customers = pl.DataFrame({
    "customer_id": [1, 2, 3, 4, 5],
    "plan": ["basic", "pro", "basic", "pro", "basic"],
    "monthly_revenue": [10, 30, 12, None, 11],
    "logins_last_14_days": [2, 15, 1, 20, 3],
})

print(customers.group_by("plan").agg(pl.col("monthly_revenue").mean()))
```

```text
plan   monthly_revenue
basic  11.0
pro    30.0
```

Note `group_by`, not pandas' `groupby` — a small but real API difference, covered below. Missing values are skipped in the mean here too, the same as pandas.

**The lazy API**, filtering and selecting columns in one query plan:

```python
lazy_query = (
    customers.lazy()
    .filter(pl.col("plan") == "pro")
    .select(["customer_id", "monthly_revenue"])
)
print(lazy_query.collect())
```

**The performance difference, measured directly** — grouping and averaging 10 million rows:

```python
import time
import numpy as np
import pandas as pd
import polars as pl

n = 10_000_000
rng = np.random.default_rng(0)
groups = rng.integers(0, 100, size=n)
values = rng.normal(0, 1, size=n)

pdf = pd.DataFrame({"group": groups, "value": values})
pldf = pl.DataFrame({"group": groups, "value": values})

t0 = time.perf_counter()
pdf.groupby("group")["value"].mean()
print(f"pandas: {(time.perf_counter() - t0)*1000:.1f} ms")

t0 = time.perf_counter()
pldf.group_by("group").agg(pl.col("value").mean())
print(f"polars: {(time.perf_counter() - t0)*1000:.1f} ms")
```

```text
pandas: 195.0 ms
polars: 31.8 ms
```

About 6x faster on this machine, for the identical aggregation over the same data.

## Where is it used?

Increasingly used as a pandas alternative or complement for medium-to-large tabular workloads in Python pipelines — exactly where the EDA and feature engineering steps from [ML Workflow](../machine-learning/ml-workflow.md) start to outgrow what a single-threaded, fully in-memory pandas DataFrame handles comfortably.

## Advantages

- **Genuinely faster for large aggregations** — measured directly above, about 6x on a 10-million-row groupby.
- **Multi-threaded by default**, with no extra configuration required, unlike pandas.
- **The lazy API optimizes an entire pipeline before running it**, catching unnecessary work automatically rather than relying on the author to hand-optimize each step.

## Limitations

- **A smaller ecosystem than pandas.** Many plotting and ML libraries expect a pandas DataFrame specifically, sometimes requiring an explicit `.to_pandas()` conversion at the boundary.
- **The eager/lazy split is an extra concept to learn**, and choosing the wrong one for a given workload — staying eager where the lazy optimizer would have helped — gives up much of Polars' advantage.
- **A newer, faster-moving library than pandas**, with a smaller base of existing tutorials and production experience to lean on when something goes wrong.

## Production considerations

- **Boundary conversions have a real cost.** Introducing Polars into part of a pipeline that otherwise expects pandas usually means a `.to_pandas()` / `.from_pandas()` conversion somewhere — that conversion cost needs to be weighed against the speedup, not ignored.
- **Lazy evaluation pays off on large, multi-step pipelines, not small one-off transformations.** For a small dataset, the eager API is simpler and the performance gap is negligible — defaulting to lazy everywhere adds complexity without a matching benefit.
- **Default multi-threading can conflict with resource-constrained environments.** A container with a strict CPU quota may see less benefit than expected, or contention, if Polars isn't explicitly configured to respect the same core limits pandas implicitly stayed within.

## Common mistakes

- **Assuming pandas method names translate directly.** Polars uses `group_by` (snake_case), not pandas' `groupby`, and other method names differ too — porting code by find-and-replace instead of checking the actual API produces `AttributeError`s.
- **Staying in eager mode for a large, multi-step pipeline** that would benefit from the lazy query planner, missing the automatic optimization that's a core reason to reach for Polars in the first place.
- **Treating "faster" as the only criterion.** The fastest tool isn't the right one if it forces an awkward conversion at every boundary with the rest of a pandas-based stack — the ecosystem cost from Limitations is a real trade-off, not a footnote.

## Interview questions

### Basic

- What's the core difference between eager and lazy execution in Polars?
- Why is Polars generally faster than pandas for large datasets?

### Intermediate

- What does it mean for Polars' lazy query planner to "push down" a filter or column selection?
- When would pandas still be the better choice over Polars, despite Polars being faster?

### Advanced

- Why might switching only part of a pipeline to Polars fail to deliver the expected speedup?
- How does Polars achieve multi-threading by default when pandas is largely single-threaded?
