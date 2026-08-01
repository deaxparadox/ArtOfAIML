# Standard Deviation

See [Variance](variance.md) for the spread measure this chapter directly builds on.

## What is it?

Standard deviation is the square root of variance. It measures the same thing — how spread out a dataset is around its mean — but back in the original units instead of squared ones.

## Why does it exist?

[Variance](variance.md) ended on its own biggest weakness: its units are squared, so "the variance in delivery time is 25 minutes²" doesn't mean much to anyone. Standard deviation exists purely to undo that — taking the square root brings the number back into the same units as the data itself, so "the standard deviation in delivery time is 5 minutes" is immediately interpretable next to "the mean delivery time is 30 minutes."

Standard deviation also carries a genuinely useful engineering shortcut for roughly normal (bell-curve-shaped) data — the **68-95-99.7 rule**: about 68% of values fall within 1 standard deviation of the mean, 95% within 2, and 99.7% within 3. That's the mathematical basis for a common real heuristic: flagging anything more than 3 standard deviations from the mean as worth investigating.

**When that rule applies, and when it doesn't:** it's specific to roughly normal distributions. On skewed or heavy-tailed data — the same kind of data where [Mean](mean.md) and [Median](median.md) already diverge — "3 standard deviations" doesn't correspond to the same probability at all, and using it as an anomaly threshold there is unreliable.

## How does it work?

Standard deviation is just `sqrt(variance)`. The population-vs-sample distinction from [Variance](variance.md) carries through directly: population standard deviation uses population variance under the square root, sample standard deviation uses sample variance (dividing by `n - 1`).

## Example

Using the same dataset as [Variance](variance.md), and confirming the same population/sample split shows up here too:

```python
import numpy as np
import statistics

data = [2, 4, 4, 4, 5, 5, 7, 9]

print(np.std(data))              # -> 2.0 (population, ddof=0 by default)
print(statistics.stdev(data))    # -> 2.138... (sample)
```

A more practical use: flagging an anomalous value with a **z-score** — how many standard deviations a value sits from the mean.

```python
import statistics

latencies = [100, 102, 98, 101, 99, 103, 97, 100, 102, 99]
mean = statistics.mean(latencies)
std = statistics.stdev(latencies)
print(mean, std)  # -> 100.1 1.912

new_value = 140
z = (new_value - mean) / std
print(z)  # -> 20.87
```

A response time of 140ms lands almost 21 standard deviations from the mean — an extreme outlier by any reasonable threshold. Notice how tight the normal range is here (std of about 1.9ms): because this system's latency is normally very consistent, even a comparatively modest 108ms reading is already more than 4 standard deviations away. The z-score threshold that counts as "anomalous" isn't a fixed number of milliseconds — it's relative to how consistent the baseline behavior actually is.

## Where is it used?

- **Anomaly detection** — flagging values with an unusually large z-score, as above.
- **Feature scaling** — `StandardScaler` (from [Feature Engineering](../machine-learning/feature-engineering.md)) transforms every feature into exactly this: a z-score, with mean 0 and standard deviation 1. Scaling a feature and detecting an anomalous value are, mathematically, the same operation.
- **Communicating spread** — standard deviation is what actually gets reported alongside a mean in almost every practical context, precisely because variance's squared units don't.
- **Statistical confidence and experiment design** — how much data you need to trust a result depends directly on how much natural spread (standard deviation) is in the underlying measurements.

## Advantages

- **Same units as the original data** — directly interpretable, unlike variance.
- **The empirical rule gives fast intuition** for how unusual a value is, when the data is roughly normal.
- **z-scores make different scales directly comparable** — the same reason `StandardScaler` uses them for feature scaling.

## Limitations

- **Inherits variance's outlier sensitivity** — squaring, then square-rooting, still weights large deviations heavily.
- **The empirical rule only holds for roughly normal data.** On skewed or heavy-tailed distributions, a "3-sigma" threshold doesn't correspond to the probability people usually assume it does.
- **A z-score is only as good as its reference mean and standard deviation.** If the window used to compute them already contains anomalies or drift, the baseline itself is wrong, and every z-score computed from it is wrong in the same direction.

## Production considerations

- **Choosing the baseline window matters as much as the threshold.** A z-score-based anomaly detector needs a "normal" reference period to compute mean and standard deviation from, and that reference needs periodic recomputation as real behavior legitimately shifts — the same drift concern already raised for retraining in [ML Workflow](../machine-learning/ml-workflow.md).
- **`StandardScaler` and anomaly detection are the same math**, deployed for different purposes — worth remembering when debugging a pipeline that does both, since a bug in one place (a contaminated baseline) silently affects the other.
- **Streaming standard deviation inherits variance's numerical stability concerns** — since it's built directly from variance, the same incremental-algorithm caution from [Variance](variance.md) applies here too.

## Common mistakes

- **Applying the 68-95-99.7 rule to skewed or heavy-tailed data** as if it applies universally — it's specific to roughly normal distributions.
- **Computing a z-score baseline from a window that already contains anomalies or drift**, contaminating the exact reference point being used to detect anomalies.
- **Confusing standard deviation with standard error** — a related but different concept describing the uncertainty of an *estimated mean* itself, not the spread of the raw data. Worth knowing they're not interchangeable, even though this chapter doesn't cover standard error in depth.

## Interview questions

### Basic

- How is standard deviation related to variance?
- Why is standard deviation usually reported instead of variance when communicating results?

### Intermediate

- What is a z-score, and how would you use it to flag an anomalous value?
- Why doesn't the 68-95-99.7 rule apply reliably to skewed data?

### Advanced

- Why might a mean/standard-deviation-based anomaly detector fail if its baseline window already contains drift or anomalies?
- How is z-score-based feature scaling mathematically the same operation as z-score-based anomaly detection?
