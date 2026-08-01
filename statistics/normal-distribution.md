# Normal Distribution

See [Standard Deviation](standard-deviation.md) for the empirical rule this chapter explains the actual shape behind.

## What is it?

The normal distribution (or Gaussian distribution) is a specific, symmetric, bell-shaped probability distribution where values cluster around the mean, with a precise mathematical relationship between distance from the mean and probability. The "68-95-99.7" empirical rule already used in [Standard Deviation](standard-deviation.md) is a direct consequence of this specific shape.

[Standard Deviation](standard-deviation.md) used the empirical rule as a practical heuristic without naming the actual shape it comes from. This chapter names it directly: the normal distribution is the specific bell curve where that rule holds exactly, defined completely by just two numbers — its mean and its standard deviation.

## Why does it exist?

[Standard Deviation](standard-deviation.md) flagged that the empirical rule "only holds for roughly normal distributions" without fully explaining why that shape shows up so often in the first place. Much of it traces back to the **Central Limit Theorem**: the sum or average of many independent, similarly-sized random factors tends toward a normal distribution, regardless of what distribution each individual factor follows. That's why so many real-world measurement processes — each shaped by many small, independent sources of variation — end up approximately normal, and why so many statistical tools (z-scores, and the hypothesis tests and confidence intervals covered in a later chapter) are built assuming it.

**When it's reasonable to assume normality, and when it isn't.** Many real processes — measurement error, averaged quantities, large samples of roughly-independent factors — really do end up approximately normal, making the assumption reasonable and its tools usable. The assumption breaks down for genuinely skewed data — income, latency, already covered in [Mean](mean.md) and [Median](median.md) — or data shaped by a small number of dominant factors rather than many independent small ones. Applying normal-distribution-based tools to visibly skewed data doesn't just introduce a little imprecision; it produces systematically wrong conclusions, as the example below shows directly.

## How does it work?

A normal distribution is fully described by exactly two parameters: its mean (center) and standard deviation (spread) — no other information changes its shape. Its probability density is highest at the mean and falls off symmetrically and smoothly on both sides. The 68-95-99.7 rule is an exact mathematical consequence of that specific shape, not a coincidence or a loose approximation — for data that only resembles normal, the rule becomes an approximation instead.

## Example

Testing the empirical rule directly on data that really is normal, and on data that clearly isn't:

```python
import numpy as np

rng = np.random.default_rng(0)

def within_k_std(data, k):
    mean, std = data.mean(), data.std()
    return np.mean(np.abs(data - mean) <= k * std)

normal_data = rng.normal(loc=100, scale=15, size=100000)
for k in [1, 2, 3]:
    print(k, within_k_std(normal_data, k))

skewed_data = rng.exponential(scale=15, size=100000)
for k in [1, 2, 3]:
    print(k, within_k_std(skewed_data, k))
```

```text
Normal distribution:  68.4%, 95.5%, 99.7%
Skewed (exponential):  86.5%, 95.0%, 98.2%
```

On true normal data, the empirical rule holds almost exactly — 68.4% within one standard deviation, matching the textbook 68% closely. On the skewed exponential distribution, the same "within one standard deviation" figure is 86.5%, nearly 20 points off from what the rule predicts. Applying a z-score threshold tuned for normal data to this skewed distribution wouldn't just be a little off — it would misjudge how unusual a value actually is by a wide margin.

## Where is it used?

The basis for z-scores and anomaly detection thresholds already covered in [Standard Deviation](standard-deviation.md), the assumption behind many classical statistical tests and confidence intervals, and a common (if sometimes wrongly assumed) default whenever "how unusual is this value" needs a quick, principled answer.

## Advantages

- **Fully described by just two numbers**, making it simple to reason about and cheap to summarize.
- **The Central Limit Theorem gives it broad real-world applicability** — many naturally aggregated or averaged quantities really do converge toward this shape.
- **A large, mature toolkit of statistical methods** is built directly on top of the normal distribution's known mathematical properties.

## Limitations

- **The assumption fails visibly on skewed data**, as the example shows directly — not a small error, a nearly 20-point miss on the very first empirical-rule threshold.
- **Being "roughly bell-shaped" isn't the same as actually being normal.** A distribution can look symmetric to the eye and still deviate enough in its tails to break tools that assume true normality.
- **Real-world data often violates the Central Limit Theorem's own conditions** — factors that aren't independent, or where one factor dominates the rest, don't reliably converge to normal no matter how many of them there are.

## Production considerations

- **Checking normality before applying a normal-distribution-based tool is a real, worthwhile step**, not a formality — a quick histogram or comparison against the empirical rule, as done above, catches the mismatch before it produces a wrong threshold in production.
- **Metrics that are naturally skewed (latency, revenue, session length) need transformation or non-parametric handling**, not a normal-distribution assumption applied out of convenience.
- **A model or monitoring system built assuming normality will silently misbehave on skewed inputs**, exactly the way the exponential distribution example above misrepresents "how unusual" a value is.

## Common mistakes

- **Assuming any roughly symmetric, bell-ish-looking distribution is normal** without actually checking it, then applying tools that depend on the assumption being true.
- **Applying z-score-based anomaly thresholds to visibly skewed metrics**, producing the exact kind of misjudgment this chapter's example demonstrates.
- **Treating the Central Limit Theorem as a guarantee that data will be normal**, rather than a statement about specific conditions (independence, comparable scale of contributing factors) that real data doesn't always meet.

## Interview questions

### Basic

- What two parameters fully describe a normal distribution?
- Why does the 68-95-99.7 rule specifically apply to normal distributions?

### Intermediate

- What does the Central Limit Theorem say, and why does it explain why normal distributions show up so often?
- Why did the empirical rule fail on the skewed distribution in this chapter's example?

### Advanced

- How would you check whether a real dataset is close enough to normal to trust a z-score-based threshold on it?
- Give an example of a real-world metric that violates the Central Limit Theorem's assumptions, and explain why.
