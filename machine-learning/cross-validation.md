# Cross-Validation

See [ML Workflow](ml-workflow.md) for the single train/test split this chapter builds on.

## What is it?

Cross-validation is a technique for estimating how well a model will generalize by repeatedly splitting the training data into different train/validation folds, training and evaluating on each split, and reporting the average — instead of relying on a single train/test split.

[ML Workflow](ml-workflow.md) introduced the train/test split as a single exam given once. Cross-validation is giving the same student several different versions of the exam — each time studying from a different subset of the material — and reporting all the scores together, so one lucky or unlucky exam doesn't stand in for the whole picture.

## Why does it exist?

A single train/test split has a real weakness: the specific way the data happened to get divided can make a model look better or worse than it actually is, purely by chance — a "lucky" test set that happens to be easy, or an "unlucky" one that happens to be hard. This matters more with a small dataset, where a single split's luck has outsized influence. Cross-validation exists to make that variability visible rather than hidden inside one number: by rotating which portion of the data is held out multiple times, the result is both a central estimate and an honest measure of how much it actually varies.

**K-fold cross-validation vs. a single train/test split, as a real decision:** a single split is faster and simpler, fine when the dataset is large enough that a single held-out portion already gives a stable estimate, and fine for quick iteration during early exploration. K-fold cross-validation costs `k` times the compute — training `k` separate models instead of one — but earns that cost specifically for smaller datasets, or whenever a trustworthy final number is actually needed for a report or a real comparison between model choices, not just a quick sanity check.

## How does it work?

Split the training data into `k` roughly equal folds. For each fold, train on the other `k - 1` folds and evaluate on the held-out one. The average of the `k` scores is the cross-validated performance estimate — and the spread across those `k` scores matters just as much as the average, since it shows how sensitive the estimate is to exactly which data ended up in which fold.

## Example

The same model, evaluated two ways: single splits with different random seeds, versus 5-fold cross-validation:

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=150, noise=0.3, random_state=0)

for seed in range(6):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    print(seed, accuracy_score(y_test, model.predict(X_test)))

cv = KFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(KNeighborsClassifier(n_neighbors=5), X, y, cv=cv)
print(scores, scores.mean(), scores.std())
```

```text
single splits: 0.889, 0.844, 0.889, 0.867, 0.889, 0.844
5-fold CV scores: [0.833 0.933 0.933 0.8   0.867]
5-fold CV mean: 0.873, std: 0.053
```

The single-split numbers already bounce around by about 4.5 points depending purely on which random seed split the data. The individual cross-validation fold scores bounce around *even more* — from 0.8 to 0.933, a 13-point spread. Cross-validation's real value isn't magically producing a more stable number than any single split would; it's refusing to report just one of those numbers as if it were the truth. The mean (0.873) is a reasonable central estimate, but the standard deviation (0.053) is just as important — it says plainly that this model's accuracy on new data could reasonably land anywhere in roughly an 87% ± 5% range, information a single split's one number simply can't communicate.

## Where is it used?

Comparing candidate models or hyperparameter settings reliably, reporting a trustworthy performance estimate rather than a single split's possibly-lucky number, and estimating how much a metric might vary on genuinely new data.

## Advantages

- **Uses the entire dataset for both training and evaluation** across the k folds, rather than permanently sacrificing one fixed portion to testing.
- **Reports variability, not just a point estimate** — the standard deviation across folds is a direct, honest measure of how much to trust the mean score.
- **Reduces (though doesn't eliminate) the risk of a misleadingly lucky or unlucky single split** driving a real decision.

## Limitations

- **Costs `k` times the compute** of a single split, since a full model gets trained and evaluated once per fold.
- **A high standard deviation across folds is itself a real, sometimes uncomfortable finding** — as the example shows, cross-validation can reveal that a model's performance is far less stable than a single split ever suggested, not just confirm a comforting number.
- **Doesn't fix data problems.** If the entire dataset is unrepresentative of what the model will see in production, no amount of clever splitting recovers that — cross-validation only estimates variance from the splitting process, not from the data's own quality.

## Production considerations

- **The reported standard deviation should influence how much confidence gets placed in a metric**, not just the mean — a mean accuracy of 87% with a 5-point standard deviation is a meaningfully different claim than the same mean with a 0.5-point standard deviation.
- **Cross-validation estimates variability from data splitting, not from production drift over time** — a stable cross-validated estimate at training time says nothing about how the model will hold up as real-world data shifts, the same concern already raised in [ML Workflow](ml-workflow.md).
- **k needs to scale sensibly with dataset size.** Too few folds on a small dataset barely improves on a single split; too many folds on a large dataset mostly adds compute cost without meaningfully tightening the estimate.

## Common mistakes

- **Reporting only the mean cross-validation score**, dropping the standard deviation and losing exactly the information that makes the estimate trustworthy or not.
- **Treating a single train/test split's score as definitive** for a small dataset, without checking how much it would have varied under a different split — precisely what this chapter's example demonstrates going wrong.
- **Assuming cross-validation validates data quality**, when it only estimates variance from how the data gets divided — a systematically biased or unrepresentative dataset stays exactly as biased no matter how it's split.

## Interview questions

### Basic

- What problem does cross-validation solve that a single train/test split doesn't?
- What does the standard deviation across cross-validation folds actually tell you?

### Intermediate

- Why can individual cross-validation fold scores vary even more than single train/test splits did, and why doesn't that undermine the technique?
- When would a single train/test split be a reasonable choice over full cross-validation?

### Advanced

- A model's 5-fold cross-validation mean accuracy is 87% with a standard deviation of 12 points. How would that change how you'd communicate this model's expected performance, compared to a report that only states the mean?
- Why doesn't a stable cross-validation estimate at training time guarantee stable performance in production?
