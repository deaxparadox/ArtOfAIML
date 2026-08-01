# Median

See [Mean](mean.md) for the salary example this chapter continues, and for why a single "typical value" summary is useful in the first place.

## What is it?

The median is the middle value of a sorted dataset — the point with exactly as many values below it as above it. For an odd number of values, it's the exact middle one; for an even number, it's the average of the two middle values.

Where the mean is a balance point pulled by every value's exact magnitude, the median only cares about order. Think of it as lining everyone up from smallest to largest value and pointing at whoever is standing exactly in the middle — it doesn't matter how far away the tallest or shortest person in line is, only where the middle of the line falls.

## Why does it exist?

The [Mean](mean.md) chapter showed a salary example where one outlier (500,000) nearly tripled the average, even though four of five people earned between 45,000 and 51,000. The median exists specifically to answer "what's typical?" in a way that doesn't get dragged around by a small number of extreme values — because it depends only on rank, not on how extreme a value is.

**When to reach for the median instead of the mean:** whenever the data is skewed or likely to contain outliers — income, house prices, latency — and the question you're answering is "what does a typical case look like?" **When the mean is still the better choice:** when the distribution is roughly symmetric (the two are close anyway), or when you specifically need a value with additive properties — total resources needed for a population is `mean × n`; there's no equivalent identity for the median, which matters for anything downstream that needs to sum or combine results.

## How does it work?

1. Sort the values.
2. If there's an odd number of values, take the middle one.
3. If there's an even number, average the two middle values.

## Example

The same salary data from the [Mean](mean.md) chapter: `[45000, 47000, 48000, 51000, 500000]`. Sorted, the middle value is `48000` — the median — compared to a mean of `138200`. The median lands right where four of the five salaries actually are; the mean doesn't.

An even-sized example makes the "average the two middle values" rule concrete, using response times in milliseconds for four requests, one of which was slow:

```python
import statistics

latencies = [120, 135, 142, 800]
print(statistics.median(latencies))       # -> 138.5
print(sum(latencies) / len(latencies))    # -> 299.25 (the mean)
```

The median (138.5) reflects what three of the four requests actually experienced. The mean (299.25) is more than double that, dragged upward by a single slow request — the same pattern as the salary example, just with latency instead of income. This isn't a coincidence: the median of a set of latencies is exactly what monitoring systems call **p50** — the value below which 50% of requests fall.

## Where is it used?

- **Latency percentiles** — p50 in a monitoring dashboard is the median of recent request latencies.
- **Income and price reporting** — "median household income," "median home price" are standard specifically because these distributions are heavily right-skewed.
- **Missing value imputation** — scikit-learn's `SimpleImputer(strategy="median")` is the standard alternative to mean imputation (covered in [Feature Engineering](../machine-learning/feature-engineering.md)) when the feature being imputed is skewed.

## Advantages

- **Robust to outliers and skew**, as both examples above show directly.
- **Matches intuitive "typical value"** better than the mean whenever a dataset has a long tail.
- **Corresponds directly to a real production concept** — it's literally the p50 percentile already used in latency monitoring.

## Limitations

- **Ignores the magnitude of extreme values entirely**, not just their outsized influence. A median can look perfectly healthy while the worst cases are arbitrarily bad — the same blind spot the [Mean](mean.md) chapter raised about tail latency, but from the opposite direction: the mean dilutes the tail into an average, the median ignores it completely.
- **Doesn't combine the way a mean does.** There's no way to compute an exact overall median from a set of group medians and their sizes — unlike the mean, which can be reconstructed from group sums and counts.
- **More expensive to compute.** Finding the median requires sorting (or a selection algorithm), while the mean is a single pass over the data — and unlike the mean's simple incremental update formula, there's no equally cheap way to update a median one new value at a time.

## Production considerations

- **Exact medians over live, high-volume streams are expensive.** Unlike the mean's constant-time incremental update, computing a true median means keeping every value or resorting to approximation algorithms — this is why real monitoring systems (t-digest, HDR Histogram) report *approximate* percentiles rather than exact ones.
- **Medians can't be merged across shards.** If ten servers each report their own median latency, there's no valid way to combine those ten numbers into the true overall median — you need either the raw data or a mergeable sketch structure built for exactly this problem.
- **Choosing median vs mean imputation is a real decision, not a default.** It should follow from whether the feature being imputed is skewed enough that the mean would misrepresent it, per the trade-off already introduced in [Feature Engineering](../machine-learning/feature-engineering.md).

## Common mistakes

- **Averaging per-shard medians to approximate a global median.** This is mathematically invalid — the same category of mistake as averaging per-group means without weighting (see [Mean](mean.md)), except there's no fix as simple as weighting; medians don't combine linearly at all.
- **Treating the median as an automatically "safer" choice than the mean in every case.** It ignores tail magnitude entirely, which is fine when you care about the typical case, but the wrong choice when the cost of extreme events is exactly what you're trying to measure (worst-case latency, financial risk exposure).
- **Reporting a median without any sense of spread.** "Median household income is $X" says nothing about how unequal the distribution above and below that point actually is.

## Interview questions

### Basic

- How do you compute the median of a dataset with an even number of values?
- Why is the median less sensitive to outliers than the mean?

### Intermediate

- Why can't you average per-server medians to get an overall median, the way you can with means?
- When would you choose mean imputation over median imputation for a missing feature, and vice versa?

### Advanced

- Why do production monitoring systems typically use approximate percentile algorithms instead of computing exact medians on live traffic?
- Give an example where relying only on the median would hide a real production problem that p95/p99 latency would catch.
