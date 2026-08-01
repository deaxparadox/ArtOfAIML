# Mean

## What is it?

The mean (or average) is the sum of a set of values divided by how many there are. It's a single number meant to represent the "center" of a dataset.

Think of it as the balance point of a seesaw: place each data point on the seesaw at its value, and the mean is the exact spot where the seesaw balances — pulled harder by values far from the center than by values close to it.

## Why does it exist?

Raw data — thousands of individual values — isn't directly actionable. A single summary number lets you compare, threshold, and monitor without inspecting every value by hand.

The mean specifically earned its default status because of a mathematical property, not just convenience: it's the single number that minimizes the total squared distance to every point in the dataset. That's not a coincidence — it's exactly why "always predict the average" is the standard baseline mentioned in [ML Workflow](../machine-learning/ml-workflow.md#common-mistakes), and why models trained to minimize squared error (like the `LinearRegression` example in [What is Machine Learning](../machine-learning/what-is-machine-learning.md)) are, in effect, learning a conditional mean.

**When the mean is the right summary, and when it isn't:** it works well when data is roughly symmetric and outlier-free. It becomes misleading on skewed data or data with extreme values, because a handful of extreme points can drag it far from where most of the data actually sits — the next chapter on the median covers the alternative for exactly that case.

## How does it work?

For a dataset of `n` values, the mean is:

```
mean = (x1 + x2 + ... + xn) / n
```

Every value contributes equally to the sum, regardless of whether it's typical or an outlier — which is precisely the source of both its usefulness and its main weakness.

## Example

```python
salaries = [45000, 48000, 51000, 47000]
mean_no_outlier = sum(salaries) / len(salaries)
print(mean_no_outlier)  # -> 47750.0

salaries_with_outlier = salaries + [500000]  # one executive salary added
mean_with_outlier = sum(salaries_with_outlier) / len(salaries_with_outlier)
print(mean_with_outlier)  # -> 138200.0
```

Adding a single value nearly triples the reported average, even though four of the five people still earn between 45,000 and 51,000. The mean isn't wrong here — 138,200 genuinely is the arithmetic average — but it's a poor answer to "what does a typical employee earn?" (The median of this same group is 48,000, much closer to what most people would call "typical" — covered in the next chapter.)

## Where is it used?

- **Baselines** — "always predict the mean" is the simplest baseline a real model has to beat, as noted in [ML Workflow](../machine-learning/ml-workflow.md#common-mistakes).
- **Feature scaling** — `StandardScaler` in [Feature Engineering](../machine-learning/feature-engineering.md) centers a feature by subtracting its mean.
- **Missing value imputation** — filling missing entries with a feature's mean is one of the most common imputation strategies (scikit-learn's `SimpleImputer(strategy="mean")`).
- **Monitoring** — average latency, average error rate, average request volume.

## Advantages

- **Cheap to compute, including incrementally** — a running mean can be updated one new value at a time without storing the full history (see Production considerations).
- **Well understood and easy to communicate** — "the average" needs no explanation to a non-technical audience.
- **Mathematically well-behaved** — its squared-error-minimizing property is exactly what makes it the natural target for regression models trained with MSE loss.

## Limitations

- **Highly sensitive to outliers**, as the salary example shows directly.
- **Misleading on skewed distributions** — income, latency, and wait times are all commonly right-skewed in the real world, and the mean of a right-skewed distribution sits above where most of the data actually is.
- **Says nothing about spread** — two datasets with the same mean can look completely different; that's what variance and standard deviation (covered in a later chapter) are for.

## Production considerations

- **"Average latency" hides tail problems.** A system can report a perfectly healthy average latency while a meaningful slice of requests take far longer — which is exactly why production monitoring reports percentiles (p50, p95, p99) instead of relying on the mean alone.
- **Computing a mean over streaming data.** Recomputing `sum(all_values) / count` from scratch on every new data point doesn't scale. The standard incremental update is `mean_new = mean_old + (x_new - mean_old) / n_new`, which updates the running mean in constant time per new value, without storing the full history.
- **Averaging averages is a common, subtle bug.** Averaging a set of per-group averages only gives the correct overall mean if every group is the same size; otherwise it silently over-weights smaller groups. The correct fix is to weight each group's average by its size, or better, recompute from the raw totals.

## Common mistakes

- **Reporting only the mean without checking the shape of the distribution**, and treating it as "the typical value" when the data is skewed enough that it isn't.
- **Averaging averages across unevenly sized groups** without weighting — a small group's mean ends up counted as heavily as a group ten times its size.
- **Using mean imputation without considering the side effect.** Filling every missing value with the same number artificially shrinks that feature's variance, which can quietly bias a downstream model — a real interaction with what [Bias vs Variance](../machine-learning/bias-vs-variance.md) covers, applied to the input data rather than the model.

## Interview questions

### Basic

- What is the formula for the arithmetic mean?
- Why is the mean sensitive to outliers?

### Intermediate

- Why might the mean be a poor summary statistic for a skewed distribution like income or latency?
- How would you compute a running mean over streaming data without storing every value?

### Advanced

- Why does mean imputation reduce a feature's variance, and why does that matter for a model trained on it?
- Why does "average latency" often fail to reflect real user experience, and what would you monitor instead?
