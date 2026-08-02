# Pandas

See [NumPy](numpy.md) for the array foundation pandas is built on top of.

## What is it?

Pandas is a Python library for working with labeled, tabular data. Its core object is the **DataFrame** — a two-dimensional table with named columns (each of which can hold a different type) and a row index, built directly on top of NumPy arrays under the hood.

If a NumPy array is a spreadsheet grid with no headers — one uniform type, no column names — pandas is that same grid with headers, row labels, and the ability to mix types across columns, the way an actual spreadsheet works.

## Why does it exist?

[NumPy](numpy.md#limitations) named this gap directly: an ndarray has no concept of named columns or row labels, and every element shares one fixed type. Real-world data almost never arrives that way — a single customer record has an integer ID, a string plan name, and a float revenue figure, all in the same row. Pandas exists to give Python native support for exactly this kind of mixed-type, labeled tabular data, so you can filter, group, and clean data by column name instead of hand-managing parallel arrays or dictionaries.

**When to reach for pandas over raw NumPy:** whenever data has meaningfully different columns or types, or needs the exploratory work described in [ML Workflow](../machine-learning/ml-workflow.md#how-does-it-work)'s EDA step — filtering, grouping, per-column missing-value handling. **When to drop back to NumPy:** at the exact point a DataFrame's columns have been reduced to a single numeric feature matrix ready for a model — scikit-learn's `LinearRegression` and the other examples throughout this handbook still expect array-like numeric input, not a labeled table.

## How does it work?

A handful of operations cover most day-to-day pandas use:

- **Boolean filtering** — selecting rows that match a condition.
- **`.groupby()`** — splitting rows into groups by a column's values, then aggregating each group.
- **Missing-value handling** — `.fillna()` / `.dropna()`, the pandas-side counterpart to the mean/median imputation covered in [Feature Engineering](../machine-learning/feature-engineering.md).

## Example

A small customer table — mixed types, one missing value:

```python
import pandas as pd
import numpy as np

customers = pd.DataFrame({
    "customer_id": [1, 2, 3, 4, 5],
    "plan": ["basic", "pro", "basic", "pro", "basic"],
    "monthly_revenue": [10, 30, 12, np.nan, 11],
    "logins_last_14_days": [2, 15, 1, 20, 3],
})

print(customers[customers["plan"] == "pro"])
```

```text
   customer_id plan  monthly_revenue  logins_last_14_days
1            2  pro             30.0                   15
3            4  pro              NaN                   20
```

```python
print(customers.groupby("plan")["monthly_revenue"].mean())
```

```text
plan
basic    11.0
pro      30.0
```

Notice the `pro` group's mean is exactly `30.0` — the single `NaN` in that group was silently excluded from the average. That's worth comparing directly to raw NumPy:

```python
print(np.mean([10, 30, 12, np.nan, 11]))     # -> nan
print(np.nanmean([10, 30, 12, np.nan, 11]))  # -> 15.75
```

Plain `np.mean` propagates the `NaN` through the entire result unless you explicitly ask for `np.nanmean`. Pandas' `.groupby().mean()` skips missing values by default, with no equivalent of NumPy's explicit "nan-aware" function needed. That default is convenient — and, as covered below, easy to rely on without noticing.

## Where is it used?

The EDA and feature engineering steps of [ML Workflow](../machine-learning/ml-workflow.md) — loading raw data, inspecting distributions, handling missing values — almost always happen in pandas before the data becomes a NumPy array for a model. More generally: joining data sources, reshaping, and aggregating are the daily work behind the observation, already made in [ML Workflow](../machine-learning/ml-workflow.md#limitations), that data work usually takes longer than model training.

## Advantages

- **Labeled, mixed-type tabular data handled natively**, unlike a raw NumPy array.
- **A large, expressive set of table operations** — filter, join, groupby, pivot — that would otherwise need to be hand-written as loops.
- **NaN-aware aggregation by default**, as shown above — reducing (though not eliminating) one category of the imputation bugs [Feature Engineering](../machine-learning/feature-engineering.md) already warned about.

## Limitations

- **Expressiveness has a performance cost.** Row-by-row Python-level operations — most commonly `.apply()` with a custom function — lose the vectorization speed that made NumPy fast in the first place, quietly falling back to a Python-level loop under the hood.
- **Not built for out-of-memory data.** A DataFrame loads entirely into memory, which becomes a hard ceiling at some scale — exactly the gap the next chapter's library, Polars, targets.
- **The default NaN-skipping is easy to rely on without noticing.** It's convenient, but it can quietly hide a missing-data problem if you're not separately checking how many values were missing in the first place.

## Production considerations

- **`pd.read_csv` loading a full file into memory doesn't scale indefinitely.** Large or growing datasets need chunked reading, a database-backed query instead of an in-memory DataFrame, or a switch to a library built for larger-than-memory data.
- **`.apply()` with a Python function is an easy-to-miss performance trap.** It looks idiomatic, but runs at Python speed per row — a fast prototype can quietly turn into a slow production job purely from data volume growth, with no code change at all.
- **Feature transformations built with pandas need to be reproduced identically at prediction time** — the same train-serving skew concern raised in [Feature Engineering](../machine-learning/feature-engineering.md), now applied to hand-written pandas preprocessing rather than a fitted scikit-learn transformer.

## Common mistakes

- **Assuming pandas and raw NumPy handle missing values the same way.** As shown above, `.groupby().mean()` silently skips `NaN`, while plain `np.mean()` on an array containing `NaN` propagates it through the entire result.
- **Using `.apply()` where a built-in vectorized operation would do the same work**, quietly losing most of the performance benefit both pandas and NumPy exist to provide.
- **Chained assignment**, e.g. `df["foo"][df["bar"] > 5] = 100`. With Copy-on-Write as the default since pandas 2.0, this now reliably does nothing to the original DataFrame — no longer the old behavior of sometimes working and sometimes not depending on internal memory layout — while raising a `ChainedAssignmentError` *warning* (not a raised, catchable exception; execution continues) to flag that the assignment was silently discarded. The fix is the same either way: `df.loc[df["bar"] > 5, "foo"] = 100` in a single step, which both works and never warns.

## Interview questions

### Basic

- What is a DataFrame, and how is it different from a NumPy array?
- Why does pandas exist when NumPy already provides arrays?

### Intermediate

- Why does `.groupby().mean()` handle missing values differently than a raw NumPy array's `.mean()`?
- Why can `.apply()` be slower than it looks, compared to an equivalent vectorized operation?

### Advanced

- At what point in an ML pipeline would you convert a pandas DataFrame back into a NumPy array, and why?
- Why does pandas struggle with datasets that don't fit in memory, and what approaches exist to handle that?
