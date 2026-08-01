# Variance

See [Mean](mean.md), [Median](median.md), and [Mode](mode.md) for the three ways to summarize where data is centered — this chapter covers how spread out it is around that center.

## What is it?

Variance measures how spread out a set of values is around its mean — specifically, the average of the squared difference between each value and the mean.

If the mean is the balance point of the data, variance answers a different question entirely: not "where is the center," but "how far, on average, does the data sit from that center." Two datasets can share the exact same mean and still look completely different, and variance is the number that tells them apart.

**A word on the name.** You may already recognize "variance" from [Bias vs Variance](../machine-learning/bias-vs-variance.md), where it described how sensitive a model's predictions are to the specific training data it saw. That's the same underlying idea — a measure of spread — just applied to a model's predictions across different training sets, instead of to a single dataset's raw values around their mean. This chapter is the base statistical definition; the ML chapter is one specific, important application of it.

## Why does it exist?

The mean alone says nothing about spread. `[45, 46, 44, 45]` and `[10, 80, 20, 70]` both average to 45, but they represent completely different realities — one is a tight cluster, the other is wildly inconsistent. Variance exists to put a number on that difference.

Deviations are squared rather than just measured as absolute distance for two concrete reasons: squaring makes every deviation positive, so they don't cancel out around the mean (a value 10 below the mean and a value 10 above it both count as "spread," not as canceling each other to zero); and squaring is smooth and differentiable, which is exactly why squared error shows up everywhere in machine learning — it's the same mathematical property that made the mean the natural target for regression models in the [Mean](mean.md) chapter.

**A real decision, not just a formula choice:** dividing by `n` (population variance) assumes you have the entire population's data. Dividing by `n - 1` (sample variance, "Bessel's correction") corrects for the fact that, when you only have a sample, the sample's own mean was estimated from that same data — which would otherwise systematically underestimate the true population variance. Almost all real engineering work uses sample variance, because you're almost always working with a sample, not a complete population.

## How does it work?

1. Compute the mean.
2. Subtract the mean from each value, and square the result.
3. Average those squared differences — dividing by `n` for population variance, or `n - 1` for sample variance.

## Example

The same-mean, different-spread claim from above, verified directly:

```python
import statistics

a = [45, 46, 44, 45]
b = [10, 80, 20, 70]

print(statistics.mean(a), statistics.mean(b))      # -> 45 45
print(statistics.variance(a), statistics.variance(b))  # -> 0.667 1233.33
```

Identical means, wildly different variance — `a` is a tight cluster, `b` isn't close to being one.

The population-vs-sample distinction isn't academic — it produces different numbers from different tools by default, and that's a real source of bugs:

```python
import numpy as np
import pandas as pd

data = [2, 4, 4, 4, 5, 5, 7, 9]

print(np.var(data))          # -> 4.0    (population, ddof=0 by default)
print(pd.Series(data).var()) # -> 4.571... (sample, ddof=1 by default)
```

NumPy and pandas disagree on the *default* — `numpy.var` assumes population variance unless you pass `ddof=1`; `pandas.Series.var` assumes sample variance unless you pass `ddof=0`. Mixing the two without noticing produces numbers that are quietly wrong, not obviously broken.

## Where is it used?

- **Feature scaling** — `StandardScaler` (from [Feature Engineering](../machine-learning/feature-engineering.md)) scales by the standard deviation, which is built directly from variance.
- **Model behavior** — the "variance" half of the bias-variance tradeoff, covered in [Bias vs Variance](../machine-learning/bias-vs-variance.md).
- **Experiment design** — how much spread is in your data directly determines how much data you need to detect a real effect with confidence.
- **Monitoring consistency**, not just averages — a metric's variance over time tells you whether a system's behavior is stable, separately from whether its average is good.

## Advantages

- **Puts an exact number on spread**, distinguishing datasets the mean alone can't tell apart.
- **Combines cleanly across independent sources** — the variance of a sum of independent quantities is the sum of their variances, a property used constantly when reasoning about combined sources of noise or error.
- **Smooth and differentiable**, which is why squared-error objectives dominate machine learning — the same property that made the mean special in the first place.

## Limitations

- **Its units are squared**, which makes it hard to interpret directly — "the variance in delivery time is 25 minutes²" doesn't mean much on its own. [Standard deviation](standard-deviation.md) exists specifically to undo this.
- **Even more sensitive to outliers than the mean.** Because deviations are squared, one extreme value contributes disproportionately more to variance than it does to the mean.
- **Treats spread above and below the mean identically.** If what actually matters is downside risk specifically — a latency spike, a financial loss — variance alone doesn't distinguish "spread that hurts" from "spread that doesn't."

## Production considerations

- **Population vs. sample variance mismatches are a real, silent bug.** As the NumPy/pandas example shows, using the wrong default produces a different number without any error — and the gap between the two formulas shrinks as `n` grows, so this bug hides comfortably in small experiments and only becomes visible (and confusing) when someone tries to reproduce a result with a different library's default.
- **Monitoring variance catches problems a mean-only dashboard misses.** A metric can have a perfectly stable average while its variance climbs — meaning the system is becoming less predictable even though "the average looks fine," the same blind spot [Mean](mean.md) already raised about tail latency, but from the consistency angle rather than the magnitude angle.
- **Naive streaming variance calculations can be numerically unstable.** Computing variance as `sum(x²)/n - mean²` is mathematically correct but prone to catastrophic cancellation in floating point for large means or small variances — production systems use numerically stable incremental algorithms (e.g. Welford's algorithm) instead.

## Common mistakes

- **Mixing population and sample variance formulas without checking which one a library defaults to** — as shown above, NumPy and pandas don't agree, and neither warns you about it.
- **Reporting variance directly to a non-technical audience.** Its squared units make it genuinely hard to interpret; standard deviation is almost always the number worth communicating instead.
- **Computing variance with the naive sum-of-squares formula in a streaming system**, risking numerical instability instead of using an incremental algorithm designed for it.

## Interview questions

### Basic

- What does variance measure?
- Why are deviations from the mean squared instead of just taken as absolute values?

### Intermediate

- What's the difference between population variance and sample variance, and why does the sample formula divide by `n - 1`?
- Why might two datasets have the same mean but very different variance, and why does that distinction matter in practice?

### Advanced

- Why is "variance" the term used in the bias-variance tradeoff, and how does that usage relate to the statistical definition in this chapter?
- Why is a naive sum-of-squares variance formula numerically risky for streaming data, and what's the standard fix?
