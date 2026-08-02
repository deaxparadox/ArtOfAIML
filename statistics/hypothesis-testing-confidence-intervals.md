# Hypothesis Testing / Confidence Intervals

See [Correlation vs Causation](correlation-vs-causation.md) for the related question of whether an observed pattern means anything at all.

## What is it?

Hypothesis testing is a formal procedure for deciding whether an observed effect in data is likely real or could plausibly be explained by random chance alone. A confidence interval is a range of plausible values for an unknown true quantity — like a true difference between two groups — built from a sample, that communicates how much uncertainty remains around a single point estimate.

A sample never perfectly reveals the truth about the whole population. Hypothesis testing and confidence intervals are two different, formal ways of being honest about exactly how much a sample-based estimate should be trusted, rather than reporting a single number as if it were certain.

## Why does it exist?

[Correlation vs Causation](correlation-vs-causation.md) showed that an observed pattern can be misleading. This chapter answers a related but distinct question: even setting causation aside, is an observed difference big enough that it probably reflects something real, or small enough that it could just be noise from a limited sample? Without this, it's tempting to treat any observed difference — "version B converted 2 points better than version A in our test" — as meaningful, when a small sample could easily produce that exact gap purely by chance, with no real underlying difference at all.

**A hypothesis test vs. a confidence interval — two views of the same uncertainty, not two unrelated tools.** A hypothesis test collapses that uncertainty into a binary decision: is this effect "statistically significant," yes or no, at some threshold. Convenient for a go/no-go decision, but it throws away information about the actual size and precision of the effect. A confidence interval keeps the full picture — the estimated effect size *and* a range reflecting how precisely it's known — which is why many practitioners now prefer reporting confidence intervals over a bare significance verdict: "B converted between -11 and +1 percentage points relative to A" is more informative than "not statistically significant."

## How does it work?

The **null hypothesis** is the default assumption that there's no real effect — version A and B convert equally well. The **p-value** is the probability of seeing an effect at least as large as the one observed, *if the null hypothesis were actually true*. A small p-value means the observed effect would be unlikely to occur from chance alone if there were truly no real effect — evidence against the null hypothesis, not proof it's false. A common threshold like `p < 0.05` decides whether to reject the null, but it's a convention, not a law of nature — and critically, `p < 0.05` does **not** mean "there's a 95% chance the effect is real." It means "if there were truly no effect, a result this extreme would show up less than 5% of the time by chance alone." A **confidence interval** is computed from the sample's variability, giving a range that would contain the true value a specified percentage of the time — typically 95% — if the same sampling process were repeated many times.

## Example

Simulating an A/B test where both groups truly convert at the exact same rate — any apparent difference is pure noise:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
true_rate = 0.10
n_per_group = 200

apparent_b_better = 0
false_significant = 0
for _ in range(2000):
    a = rng.binomial(1, true_rate, n_per_group)
    b = rng.binomial(1, true_rate, n_per_group)
    if (b.mean() - a.mean()) >= 0.02:
        apparent_b_better += 1
    _, p = stats.ttest_ind(a, b)
    if p < 0.05:
        false_significant += 1

print(apparent_b_better / 2000)   # -> 0.2575
print(false_significant / 2000)  # -> 0.052
```

```text
B appears >=2 points better than A, purely by chance: 25.8%
formal t-test falsely "significant" despite no real difference: 5.2%
```

With genuinely identical conversion rates, B looks meaningfully better than A purely from sampling noise more than a quarter of the time. A naive "B looked better, ship it" decision would be wrong roughly 1 in 4 times here. The formal hypothesis test does much better — it's falsely significant only about 5.2% of the time, matching its designed `p < 0.05` false-positive rate almost exactly. It doesn't eliminate false positives (that's what the 5% threshold accepts by design), but it cuts the false-alarm rate by a factor of five compared to eyeballing the raw numbers.

A confidence interval on one specific run makes the honest conclusion directly visible:

```python
rng = np.random.default_rng(0)
a = rng.binomial(1, true_rate, 200)
b = rng.binomial(1, true_rate, 200)
diff = b.mean() - a.mean()
se = np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
print(diff, diff - 1.96*se, diff + 1.96*se)
```

```text
observed difference: -0.05
95% confidence interval: (-0.115, 0.015)
```

The point estimate alone (-0.05) looks like B underperformed by 5 points. The confidence interval spans from -0.115 to +0.015 — it includes zero, honestly signaling that a true difference of exactly zero is entirely consistent with this data. Reporting only the point estimate would have overstated confidence in a difference that isn't actually there.

## Where is it used?

A/B testing and experiment analysis, deciding whether a model improvement is real or within noise, and any claim of the form "X is better than Y" backed by a limited sample rather than complete population data.

## Advantages

- **Distinguishes a real effect from sampling noise**, as the example shows directly — without it, a quarter of null-effect experiments here would have looked like real wins.
- **Confidence intervals communicate precision, not just a point estimate**, making it visible when "no real difference" is still a plausible explanation.
- **A shared, well-understood convention** (`p < 0.05`, 95% confidence intervals) that makes results comparable across studies and teams.

## Limitations

- **A p-value threshold doesn't eliminate false positives — it accepts a known rate of them.** The 5.2% false-significance rate in the example is exactly what `p < 0.05` is designed to allow, not a flaw in the method.
- **Statistical significance isn't the same as practical significance.** A large enough sample can make a tiny, practically meaningless difference "statistically significant."
- **The p-value is one of the most commonly misinterpreted numbers in statistics** — as noted above, it is not "the probability the null hypothesis is true," a mistake made often enough to be worth stating twice.

## Production considerations

- **Running many simultaneous hypothesis tests multiplies the false-positive rate.** Testing 20 independent metrics at `p < 0.05` each means a false "significant" result somewhere is likely, not unlikely, even with no real effects anywhere.
- **Sample size needs to be planned before running an experiment**, not adjusted after seeing early results — stopping a test as soon as it looks significant inflates the false-positive rate well above the nominal threshold.
- **A confidence interval's width should directly inform confidence in a decision** — a wide interval that still crosses zero, as in the example above, means the data doesn't yet support acting on the observed difference.

## Common mistakes

- **Treating any observed difference as meaningful**, without checking whether it's within the range plausible from chance alone — exactly the 25.8% false-alarm rate this chapter's example demonstrates.
- **Misreading a p-value as "the probability the effect is real."** It's the probability of the observed data (or more extreme) under the assumption there's no effect — a different claim entirely.
- **Peeking at results and stopping an experiment early** the moment it looks significant, inflating the true false-positive rate above the stated threshold.

## Interview questions

### Basic

- What does a p-value actually represent?
- What does a 95% confidence interval mean?

### Intermediate

- Why did the naive comparison in this chapter's example show B "winning" 25.8% of the time when there was no real difference?
- Why is a confidence interval that includes zero an important result, not a non-result?

### Advanced

- Why does running many simultaneous hypothesis tests increase the overall false-positive rate, and how would you account for that?
- Why can stopping an A/B test as soon as it becomes "significant" invalidate the test's statistical guarantees?
