# Bayes Theorem

See [Probability](probability.md) for conditional probability, which this chapter builds on directly.

## What is it?

Bayes' theorem is a formula for correctly flipping a conditional probability around — going from `P(B | A)` to `P(A | B)` — while properly accounting for how likely `A` was in the first place:

```
P(A | B) = P(B | A) × P(A) / P(B)
```

- `P(A)` — the **prior**: what you believed about A before seeing any evidence.
- `P(B | A)` — the **likelihood**: how likely the evidence is, assuming A is true.
- `P(A | B)` — the **posterior**: your updated belief about A, after seeing the evidence.

Think of the prior as your best guess before any evidence arrives, and Bayes' theorem as the precise, principled amount to move that guess once new evidence shows up — no more, no less.

## Why does it exist?

[Probability](probability.md) ended on a specific, easy-to-make mistake: assuming the independence rule applies when it doesn't. Bayes' theorem exists to fix an even more common and more consequential mistake — confusing `P(A | B)` with `P(B | A)`, as if they were the same number.

They very often aren't. Knowing "how likely a symptom is, given a disease" (something you can measure directly from clinical data) is not the same as knowing "how likely the disease is, given the symptom" (what a patient actually wants to know). Bayes' theorem is the exact formula for converting one into the other, and it does so by explicitly bringing in the prior — the piece that naive intuition tends to forget entirely.

**When this matters most, as an engineering decision:** whenever the base rate of what you're detecting is far from 50/50 — fraud, rare disease, rare hardware defects. In those cases, treating a positive signal at face value, without correcting for how rare the thing being detected actually is, leads to systematically wrong conclusions. This is the same calibration concern raised in [Probability](probability.md), made precise and computable.

## How does it work?

When there are exactly two possibilities (`A` and `not A`), the denominator `P(B)` expands using the law of total probability:

```
P(B) = P(B | A) × P(A) + P(B | not A) × P(not A)
```

Which gives the full working formula:

```
P(A | B) = [P(B | A) × P(A)] / [P(B | A) × P(A) + P(B | not A) × P(not A)]
```

## Example

**Flipping the office-hours example from [Probability](probability.md).** That chapter established `P(pass | attended) = 0.75`, `P(attended) = 8/20`, and `P(pass) = 0.55` directly from counts. Bayes' theorem gets the reverse direction — `P(attended | pass)` — from those same numbers, without needing to recount anything:

```python
from fractions import Fraction

p_attended = Fraction(8, 20)
p_pass_given_attended = Fraction(6, 8)
p_pass = Fraction(11, 20)

p_attended_given_pass = (p_pass_given_attended * p_attended) / p_pass
print(p_attended_given_pass)  # -> 6/11 (0.5455)
```

That matches direct counting exactly: of the 11 students who passed, 6 attended office hours, so `6/11`. Bayes' theorem isn't a different answer — it's a way to get that answer when you only have the conditional probabilities, not the raw counts to check against directly.

**The classic, dramatic illustration: a "99% accurate" medical test.**

```python
from fractions import Fraction

p_disease = Fraction(1, 100)              # 1% prevalence (the prior)
p_no_disease = 1 - p_disease
p_pos_given_disease = Fraction(99, 100)   # 99% sensitivity
p_pos_given_no_disease = Fraction(5, 100) # 5% false-positive rate

p_positive = p_pos_given_disease * p_disease + p_pos_given_no_disease * p_no_disease
p_disease_given_positive = (p_pos_given_disease * p_disease) / p_positive

print(float(p_positive))               # -> 0.0594
print(float(p_disease_given_positive)) # -> 0.1667
```

Despite a test that correctly identifies 99% of true cases, a positive result means only a **16.7%** chance of actually having the disease. Because the disease is rare, the 5% false-positive rate applied to the large healthy population produces far more false positives in absolute terms than the true positives from the small infected population. The test's accuracy figure alone was never the whole story — the prior did most of the work in that final number.

## Where is it used?

- **[Naive Bayes](../machine-learning/naive-bayes.md) classifiers** — an entire, still widely used model family built directly on this formula, using the independence assumption flagged in [Probability](probability.md) (hence "naive").
- **Spam filtering** — the original, classic application of Bayesian reasoning to text classification.
- **Fraud and rare-event detection** — where the base rate of the event being detected is low, exactly the setting the medical test example illustrates.
- **Belief updating over time** — a posterior from one round of evidence becomes the prior for the next, the basis for continuously updating a system's confidence as new signals arrive.

## Advantages

- **Gives an exact, principled way to flip a conditional probability**, instead of relying on intuition — which the medical test example shows can be dramatically wrong.
- **Forces the prior into the calculation explicitly**, which naive interpretation of a single conditional probability tends to skip entirely.
- **Composable over time** — a posterior can become the next prior, giving a clean framework for updating a belief as evidence accumulates, rather than recomputing from scratch each time.

## Limitations

- **Requires a prior, and a wrong prior gives a wrong posterior even with a perfectly correct formula.** In practice, the true base rate (of fraud, of a disease, of any rare event) is often itself uncertain or contested.
- **Naive Bayes' independence assumption is usually unrealistic.** Features are frequently correlated in practice, and applying the formula as if they aren't can produce probability estimates that are poorly calibrated, even when the classifier's overall rankings are still useful.
- **The evidence term `P(B)` gets expensive with many hypotheses.** As the number of possible explanations for the evidence grows, computing that denominator exactly becomes computationally expensive — a real, practical reason exact Bayesian inference is often replaced with approximations in more complex models.

## Production considerations

- **Rare-event priors are exactly where naive thresholds fail**, as the medical test example shows numerically. Any system flagging a rare event from an imperfect signal — fraud, defects, disease — needs this base-rate correction explicitly, or most flagged cases will be false positives, quietly overwhelming a review team's capacity to act on them.
- **An unversioned, silently drifting prior breaks reproducibility.** If a system's "belief" about a user or entity updates over time, there needs to be a clear record of what the prior was at any given moment — otherwise a past posterior probability can't be reproduced or debugged later, the same concern [ML Workflow](../machine-learning/ml-workflow.md) already raised about reproducibility generally.
- **Naive Bayes is still used in production despite its unrealistic assumption** — for computational efficiency and because it performs surprisingly well in practice, a real example of a "wrong" simplifying assumption still being genuinely useful.

## Common mistakes

- **Confusing `P(A | B)` with `P(B | A)`** — treating "the test is 99% accurate" as if it directly means "a positive result is 99% likely to be correct," which the medical test example shows can be off by a wide margin. This specific confusion has a name in law and forensics: the prosecutor's fallacy.
- **Ignoring the prior entirely when interpreting a positive signal**, alert, or model flag — the single most common version of the mistake above.
- **Assuming Naive Bayes' independence assumption doesn't matter because the classifier "still works okay."** Its rankings can be useful while its raw probability outputs are poorly calibrated — which matters a great deal the moment those probabilities are used for a threshold-based production decision.

## Interview questions

### Basic

- What is Bayes' theorem, and what do prior, likelihood, and posterior mean?
- Why can `P(A | B)` and `P(B | A)` be very different numbers?

### Intermediate

- A test is 99% accurate, but the condition it detects is rare. Why might a positive result still mean the person probably doesn't have the condition?
- Why is Naive Bayes called "naive," and what does that assumption cost in practice?

### Advanced

- How would you decide whether a model's raw probability output can be trusted as a true posterior probability, versus treated only as a useful ranking score?
- In a production fraud-detection system with a very low fraud prior, how does Bayes' theorem change how you'd interpret and act on model alerts?
