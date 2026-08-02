# Drift Monitoring

See [Observability](observability.md) for the anomaly-detection mechanism this chapter applies to a slower, statistical kind of problem.

## What is it?

Drift monitoring is the practice of detecting when a model's real-world input data (**data drift**) or the actual relationship between inputs and the correct output (**concept drift**) changes enough over time that the model's predictions are likely to have degraded, even though nothing about the model itself changed.

[Observability](observability.md)'s own example detected one anomalous request — a single latency spike. Drift monitoring detects a different, slower kind of problem: not one bad request, but a gradual shift in the statistical properties of the data flowing through the system over time, the same kind of change [Bias vs Variance](../machine-learning/bias-vs-variance.md) and [Standard Deviation](../statistics/standard-deviation.md) already touched on — a model's training-time assumptions quietly stop matching reality.

## Why does it exist?

[ML Workflow](../machine-learning/ml-workflow.md#production-considerations) already raised the question of a retraining trigger, and [Bias vs Variance](../machine-learning/bias-vs-variance.md#production-considerations) already flagged that "a growing train/test gap in production can mean the real-world data distribution has moved since training." Drift monitoring exists to turn that vague concern into a concrete, measured signal: instead of waiting for a business metric to visibly degrade — which might take a long time to notice, or be hard to attribute to the model specifically — drift monitoring directly compares the current input data's statistical properties against the training data's, flagging a meaningful shift before it necessarily shows up as an obvious problem elsewhere.

**Data drift vs. concept drift — a real, distinct pair of problems requiring different detection.** Data drift is a change in the distribution of input features — customers are suddenly older on average than the training data — detectable purely by comparing input statistics over time, without needing to know the correct label for new data at all. Concept drift is a change in the actual relationship between inputs and outputs — the same customer profile that used to predict "will churn" now predicts "won't churn," because market conditions changed — genuinely harder to detect, since it usually requires eventually observing the true outcome to notice the relationship itself has shifted, not just the inputs.

## How does it work?

**Data drift detection** compares the statistical distribution of live input features against the training data's distribution, flagging a feature whose live distribution has shifted meaningfully from what the model was trained on. **Concept drift detection** requires eventually observing the true outcome for live predictions and comparing the model's accuracy over time — a genuine performance drop, even with stable input data, signals the input-output relationship itself has changed. Both typically feed into a retraining trigger: a defined threshold that, once crossed, kicks off retraining rather than waiting indefinitely.

## Example

Comparing a model's live input distribution against its training distribution, using the exact z-score mechanism already verified in [Standard Deviation](../statistics/standard-deviation.md):

```python
import statistics

training_ages = [28, 34, 41, 25, 38, 45, 30, 33, 29, 42, 36, 27, 31, 39, 44]
live_ages_week1 = [30, 35, 40, 27, 37, 44, 31, 32, 30, 41]  # similar to training
live_ages_week8 = [52, 58, 61, 55, 60, 57, 54, 59, 56, 62]  # genuinely shifted, much older

train_mean = statistics.mean(training_ages)
train_std = statistics.stdev(training_ages)

for label, live_data in [("week 1", live_ages_week1), ("week 8", live_ages_week8)]:
    live_mean = statistics.mean(live_data)
    z = (live_mean - train_mean) / train_std
    print(label, live_mean, z)
```

```text
week 1: live mean age=34.7, z-score=-0.02
week 8: live mean age=57.4, z-score=3.49
```

Week 1's live data closely tracks the training distribution — a z-score near zero, no meaningful drift. By week 8, the live customer age distribution has shifted dramatically — a z-score of 3.49, well past the 3-sigma threshold already used as a standard anomaly flag in [Standard Deviation](../statistics/standard-deviation.md) and [Observability](observability.md#example). A model trained on the original age distribution is now serving predictions for a genuinely different customer population than the one it learned from — exactly the kind of shift that degrades a model's accuracy silently, without a single anomalous request ever appearing.

## Where is it used?

Any production model expected to serve consistently over weeks or months, especially ones facing changing user populations, seasonal effects, or shifting market conditions — the same models where [ML Workflow](../machine-learning/ml-workflow.md)'s own retraining question needs a concrete, monitored answer rather than a guess.

## Advantages

- **Detects a degrading model before a business metric visibly suffers**, as the example shows directly — the shift is measurable well before it necessarily shows up elsewhere.
- **Data drift detection needs no labeled outcome data at all**, unlike concept drift or direct accuracy monitoring, making it available immediately rather than after outcomes are eventually observed.
- **Gives a concrete, threshold-based trigger for retraining**, replacing a vague "retrain periodically" schedule with a measured signal.

## Limitations

- **Data drift alone doesn't prove the model's accuracy has actually degraded** — inputs can shift without the underlying input-output relationship changing meaningfully; concept drift is what actually confirms a real accuracy problem.
- **Concept drift is genuinely harder to detect**, since it requires the true outcome to eventually be observed — for some tasks, that observation lag can be long enough that real damage accumulates before it's caught.
- **A drift threshold is a real decision, not a given** — too sensitive, and normal seasonal variation triggers unnecessary retraining; too lax, and a real shift goes unnoticed for too long.

## Production considerations

- **The reference distribution used for drift comparison needs to be a clean, representative training snapshot**, the same baseline-contamination concern already raised in [Observability](observability.md) and [Standard Deviation](../statistics/standard-deviation.md).
- **A detected drift needs a defined response**, not just an alert — retraining, per [ML Workflow](../machine-learning/ml-workflow.md), or a rollback via [Canary Deployment](canary-deployment.md) if a recent model change is the actual cause rather than the data itself.
- **Multiple features drifting simultaneously is a stronger signal than one feature alone**, and monitoring should account for that rather than treating every feature's drift check independently and in isolation.

## Common mistakes

- **Monitoring only for data drift and assuming that's equivalent to monitoring model accuracy** — a model can serve a drifted input distribution and still perform fine, or serve a stable distribution and still be failing due to concept drift.
- **Setting a drift threshold without accounting for normal seasonal or cyclical variation**, triggering unnecessary retraining on expected, harmless fluctuation.
- **Waiting for a business metric to visibly degrade before checking for drift**, when the same signal was measurable well earlier, as this chapter's example shows directly.

## Interview questions

### Basic

- What's the difference between data drift and concept drift?
- Why can data drift be detected without needing to know the correct label for new data?

### Intermediate

- Why might a model's input distribution stay stable while concept drift still causes real accuracy problems?
- Why does the z-score threshold used in this chapter's example need to be chosen deliberately, rather than left at a default?

### Advanced

- Design a drift monitoring system for a model where the true outcome (needed to detect concept drift) isn't observable for several months after a prediction is made. What would you monitor in the meantime?
- A drift alert fires, but the model's actual accuracy hasn't measurably changed. How would you determine whether the alert is a false positive or an early warning of a problem not yet visible in accuracy?
