# Correlation vs Causation

## What is it?

Correlation measures how strongly two variables move together. Causation means one variable actually produces a change in the other. Two variables can be strongly correlated with no causal relationship between them at all — a distinction that matters enormously for how a correlation number should actually be used.

A correlation coefficient is a single number describing how two variables move together, but like [the mean](mean.md), it can't tell you why. Two variables can move together because one causes the other, because a third hidden variable causes both, or by pure coincidence — the number alone can't distinguish between these.

## Why does it exist?

Correlation is easy to compute directly from data — it's just a summary statistic on two variables. Causation requires a much stronger claim: that intervening on one variable would actually change the other. Confusing the two is one of the most common, consequential mistakes in applying statistics to real decisions. A business might see that customers who use a feature also spend more, and assume promoting the feature will increase spending — when in reality, customers who were already going to spend more are simply also the kind of customers likely to try new features. A hidden common cause, not a causal link from feature to spending.

**When correlation is enough, and when it isn't, is a real decision.** If the goal is prediction only — this feature's usage is a strong predictor of future spending, so target these customers for something else — correlation is genuinely sufficient; you don't need to know *why* two things move together to use that relationship for forecasting. If the goal is intervention — change X specifically to cause a change in Y — correlation alone is not sufficient evidence, and requires either a controlled experiment or careful reasoning about confounders.

## How does it work?

A correlation coefficient (commonly Pearson's) is a number between -1 and 1 measuring the strength and direction of a *linear* relationship between two variables: +1 is perfectly positively linearly related, -1 perfectly negatively, 0 means no linear relationship. A correlation of 0 doesn't mean no relationship at all — only no *linear* one; two variables can have a strong non-linear relationship and still show zero linear correlation. A **confounding variable** is a third factor that influences both variables being studied, creating a correlation between them with no direct causal link at all.

## Example

The classic ice-cream-and-drowning story, verified directly: both driven by a hidden confounder, temperature.

```python
import numpy as np

rng = np.random.default_rng(0)
n = 500

temperature = rng.uniform(0, 40, n)
ice_cream_sales = 50 + 4 * temperature + rng.normal(0, 20, n)
drowning_incidents = 1 + 0.3 * temperature + rng.normal(0, 2, n)

raw_corr = np.corrcoef(ice_cream_sales, drowning_incidents)[0, 1]
print(raw_corr)  # -> 0.798

def residualize(y, x):
    slope, intercept = np.polyfit(x, y, 1)
    return y - (slope * x + intercept)

ice_cream_resid = residualize(ice_cream_sales, temperature)
drowning_resid = residualize(drowning_incidents, temperature)
print(np.corrcoef(ice_cream_resid, drowning_resid)[0, 1])  # -> -0.021
```

Ice cream sales and drowning incidents are strongly correlated — 0.798, a strong relationship by any standard. Neither one causes the other; both are driven by temperature. Once temperature's effect is removed from each variable — leaving only what's left after accounting for it — the correlation between what remains drops to -0.021, essentially zero. The entire apparent relationship between ice cream and drownings was temperature, not a direct connection between the two.

## Where is it used?

Interpreting A/B test results and business metrics correctly, feature selection for predictive models (where correlation genuinely is enough, since prediction doesn't require causation), and any decision that involves "if we change X, will Y change" rather than "does X predict Y."

## Advantages

- **Cheap and fast to compute** directly from observational data, with no experiment required.
- **Fully sufficient for pure prediction tasks**, where the goal is forecasting Y from X, not understanding why they're related.
- **A useful first signal for investigation** — a correlation worth explaining is often where a genuine causal story, or a genuine confounder, gets discovered.

## Limitations

- **Says nothing about direction or mechanism.** A correlation between X and Y is equally consistent with X causing Y, Y causing X, or neither causing the other, as the ice-cream example shows directly.
- **A confounder can produce an arbitrarily strong correlation with zero real connection between the two variables being examined** — 0.798, in this chapter's example, entirely explained away by a single hidden factor.
- **Only captures linear relationships by default.** A real, strong non-linear relationship between two variables can show a Pearson correlation near zero.

## Production considerations

- **Acting on a correlation as if it were causal is a decision with real cost** — a feature promoted because it correlates with higher spending won't actually increase spending if the true driver was a confounder, wasting the investment on the wrong lever.
- **Confounders are often not obvious in advance.** Discovering one, as in this chapter's example, usually requires deliberately checking for a plausible hidden shared cause, not something a correlation coefficient reveals on its own.
- **Feature selection based on correlation alone is fine for prediction, risky for intervention** — a model that uses a confounded feature can still predict well, but a business decision built on the same feature assuming causation can fail entirely.

## Common mistakes

- **Treating any strong correlation as evidence of a causal relationship**, without checking for an obvious or plausible confounder first.
- **Assuming zero correlation means no relationship at all**, when it only rules out a linear one.
- **Using a correlational finding to justify an intervention** — changing X to influence Y — without the experimental or causal reasoning that claim actually requires.

## Interview questions

### Basic

- What's the difference between correlation and causation?
- What does a correlation coefficient of 0 actually tell you, and what doesn't it tell you?

### Intermediate

- Why can two variables be strongly correlated with no causal relationship between them?
- Why is correlation sufficient for a purely predictive task but not for deciding whether to intervene on a variable?

### Advanced

- You find a strong correlation between two business metrics and want to recommend an intervention based on it. What would you need to establish first, and why isn't the correlation itself enough?
- How would you go about identifying a plausible confounder for an observed correlation you suspect isn't causal?
