# Probability

## What is it?

Probability is a number between 0 and 1 that quantifies how likely an event is — 0 meaning impossible, 1 meaning certain. It's the mathematical language for reasoning about uncertainty precisely, instead of relying on gut feeling.

You've already seen probability in practice: [Types of Machine Learning](../machine-learning/types-of-machine-learning.md) showed `LogisticRegression.predict_proba()` outputting `[0.5, 0.5]` at a decision boundary. That output *is* a probability, in exactly the sense this chapter defines — this chapter is the math underneath that number.

## Why does it exist?

Real systems constantly have to act under uncertainty: a spam filter can't be certain an email is spam, a fraud model can't be certain a transaction is fraudulent. Probability exists to let that uncertainty be reasoned about numerically and combined from multiple pieces of evidence in a principled way, rather than through ad-hoc rules.

**A real decision this enables: predicted class vs. predicted probability.** It's common to only look at a model's hard `predict()` output and ignore `predict_proba()` — but that throws away real information. [Types of Machine Learning](../machine-learning/types-of-machine-learning.md#example) showed exactly why this matters: at 4.5 hours studied, the model predicted class 0, but the probability was almost exactly 50/50 — genuine uncertainty a hard label alone completely hides. Use the hard label when you need an automatic yes/no decision; use the actual probability when you need to rank, prioritize, or make a risk-based decision, where "0 or 1" isn't enough information.

## How does it work?

A few rules cover most practical probability reasoning:

- **Range**: every probability is between 0 and 1, and the probabilities of all possible outcomes sum to 1.
- **Complement**: `P(not A) = 1 - P(A)`.
- **Addition** (for mutually exclusive events): `P(A or B) = P(A) + P(B)`, if A and B can't both happen.
- **Multiplication** (for independent events): `P(A and B) = P(A) × P(B)`, if A and B don't affect each other.
- **Conditional probability**: `P(A | B)` — the probability of A, *given that B has already happened*. This is the concept the next chapter, Bayes' Theorem, builds on directly.

## Example

Conditional probability, computed directly from counts — no formula needed beyond division. Out of 20 students: 8 attended office hours (6 of whom passed), and 12 didn't (5 of whom passed).

```python
from fractions import Fraction

total = 20
attended, attended_pass = 8, 6
not_attended, not_attended_pass = 12, 5

p_pass = Fraction(attended_pass + not_attended_pass, total)
p_pass_given_attended = Fraction(attended_pass, attended)
p_pass_given_not_attended = Fraction(not_attended_pass, not_attended)

print(float(p_pass))                    # -> 0.55
print(float(p_pass_given_attended))     # -> 0.75
print(float(p_pass_given_not_attended)) # -> 0.4166...
```

Overall, 55% of students passed. But that overall number hides a real difference: 75% of students who attended office hours passed, versus about 42% of those who didn't. `P(pass | attended)` is a completely different, more useful number than `P(pass)` alone — it's the same reason a model's probability output, conditioned on the specific input it's given, is more useful than one overall base rate.

## Where is it used?

- **Every probabilistic model output** — the `predict_proba()` values seen throughout [Types of Machine Learning](../machine-learning/types-of-machine-learning.md).
- **Ranking and prioritization** — recommendation systems ranking by probability of a click or purchase, fraud systems ranking transactions by probability of fraud.
- **Risk-based decisions** — deciding an action based on how likely, not just whether, something is true.
- **Experiment analysis** — reasoning about how likely an observed effect is to be real versus due to chance.

## Advantages

- **A principled way to combine uncertain evidence**, instead of ad-hoc rules.
- **Preserves more information than a binary decision** — a ranking by probability lets you prioritize, a yes/no label doesn't.
- **The same math applies everywhere** — the rules above work identically whether you're reasoning about coin flips or a neural network's output layer.

## Limitations

- **Only covers uncertainty within a defined space of outcomes.** It handles "which of these known possibilities will happen" well, but says nothing about a possibility that was never represented at all — a fraud model can only assign a probability to fraud patterns it has some notion of, not to a genuinely novel pattern it's never seen anything like.
- **A model's predicted probability isn't automatically correct.** A model outputting `P(spam) = 0.9` doesn't guarantee that 90% of such predictions are actually spam, unless the model is calibrated — treating raw model output as a trustworthy probability is a common and risky assumption.
- **The independence assumption behind the multiplication rule often doesn't strictly hold**, and applying it anyway produces a subtly wrong combined probability — worth remembering going into the next chapter, where this exact assumption gets a name of its own.

## Production considerations

- **Calibration has to be checked, not assumed.** Before a fraud model's output is used for something like "auto-block anything above 90%," its probabilities need to be validated against real-world outcome rates (a calibration curve) — an uncalibrated score can look like a probability without behaving like one.
- **Threshold choice is an engineering decision with real trade-offs**, not a default. A lower flagging threshold catches more true positives at the cost of more false positives; the right threshold depends on the actual, often asymmetric, cost of each error type in the specific business context — not a universal 0.5 cutoff.
- **Combining probabilities across models or signals requires checking independence.** Naively multiplying probabilities from correlated signals (two models both influenced by the same underlying factor) overstates how confident the combined result really is.

## Common mistakes

- **Reading only `predict()` and ignoring `predict_proba()`** — the same mistake the 4.5-hour example in [Types of Machine Learning](../machine-learning/types-of-machine-learning.md#example) was built to expose.
- **Assuming independence between correlated events or features** when combining probabilities, producing a combined estimate that's confidently wrong.
- **Treating a model's raw output score as a genuine probability without checking calibration** — a common, and often invisible, source of bad production decisions.

## Interview questions

### Basic

- What is the valid range for a probability, and what do 0 and 1 represent?
- What's the difference between `P(A)` and `P(A | B)`?

### Intermediate

- Why can't you always multiply two probabilities together to get the probability of both events happening?
- Why might a model's `predict_proba()` output not be a trustworthy probability?

### Advanced

- What does it mean for a model's probability outputs to be "well-calibrated," and why does that matter for using them in production decisions?
- How would you choose a probability threshold for an automated action, given that false positives and false negatives have different costs?
