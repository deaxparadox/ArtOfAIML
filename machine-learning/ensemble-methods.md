# Ensemble Methods

See [Supervised Learning](supervised-learning.md) for the single-tree baseline this chapter improves on, and [Bias vs Variance](bias-vs-variance.md) for the train/test gap diagnostic used to measure that improvement.

## What is it?

Ensemble methods combine multiple models into one prediction, typically outperforming any single model in the ensemble alone — using either many independent models trained in parallel and averaged (**bagging**), or a sequence of models each correcting the previous one's mistakes (**boosting**).

An ensemble is like asking several independent experts for their opinion and combining their answers, rather than trusting a single expert's judgment alone. Errors specific to one expert's blind spots tend to cancel out when averaged across several — as long as the experts aren't all making the same mistake.

## Why does it exist?

A single decision tree can overfit easily on its own, and [Bias vs Variance](bias-vs-variance.md) established the underlying tension between a model's flexibility and its stability. Ensemble methods exist to get the best of both: combine many individually imperfect models in a way that reduces the weaknesses of any one of them, without needing a single "perfect" model to begin with.

**Bagging vs. boosting is a real, structural difference in how models get combined, not two flavors of the same idea.** **Bagging** (Bootstrap Aggregating, e.g. Random Forest) trains many independent models in parallel, each on a different random subset of the training data, then averages their predictions. Since each model overfits differently — to its own random subset — averaging cancels out the individual overfitting; bagging is primarily a variance-reduction technique. **Boosting** (e.g. Gradient Boosting) trains models sequentially, where each new model specifically focuses on correcting the previous ensemble's mistakes. This reduces bias — a sequence of individually limited models can together represent a complex relationship none of them could alone — but can increase variance if taken too far, since fitting too many rounds eventually starts fitting noise in the remaining errors.

## How does it work?

**Bagging** samples the training data with replacement to create several different training sets, trains one model per sample independently, then averages the predictions (regression) or votes (classification). **Boosting** trains an initial simple model, computes its errors, trains the next model specifically to correct those errors, and repeats — combining every model's contribution into a final prediction.

## Example

The same non-linearly-separable data from [Supervised Learning](supervised-learning.md#example), comparing a single tree against both ensemble strategies:

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=300, noise=0.3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

models = {
    "Single DecisionTree": DecisionTreeClassifier(random_state=0),
    "RandomForest (bagging)": RandomForestClassifier(n_estimators=100, random_state=0),
    "GradientBoosting (boosting)": GradientBoostingClassifier(n_estimators=100, random_state=0),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(name, accuracy_score(y_train, model.predict(X_train)), accuracy_score(y_test, model.predict(X_test)))
```

```text
Single DecisionTree:         train=1.000, test=0.844
RandomForest (bagging):      train=1.000, test=0.933
GradientBoosting (boosting): train=1.000, test=0.900
```

All three memorize the training data perfectly — a train accuracy of 1.000 across the board, exactly the "warning sign" [Bias vs Variance](bias-vs-variance.md#common-mistakes) already flagged. The test accuracy is where the real difference shows up: the single tree's train/test gap is the largest of the three, while Random Forest closes it the most, gaining nearly 9 points of test accuracy over the single tree with no other change to the underlying model — just many of them, trained on different bootstrap samples and averaged.

## Where is it used?

Random forests and gradient boosting are among the most commonly used algorithms for structured, tabular data in production — fraud detection, credit scoring, ranking systems, and most Kaggle-style competitions on tabular data.

## Advantages

- **Directly reduces overfitting from a single model**, as the example shows — a meaningful test-accuracy gain with no new features or data.
- **Bagging parallelizes trivially**, since each model in the ensemble trains independently of the others.
- **Boosting can reduce bias that no single weak model in the sequence could on its own**, building up a complex relationship incrementally.

## Limitations

- **An ensemble is less interpretable than a single model.** A single decision tree's decision path is easy to trace; a forest of a hundred trees, or a boosted sequence of them, isn't.
- **Boosting can overfit if run for too many rounds** — the same variance risk [Bias vs Variance](bias-vs-variance.md) already covered, just introduced by too many boosting iterations instead of too much model complexity directly.
- **More models means more compute and memory**, both to train and to serve — an ensemble's prediction cost is the sum of its members' costs, not free.

## Production considerations

- **Inference latency scales with ensemble size** — a hundred-tree random forest is meaningfully slower to score a single request than one tree, a real cost that has to be weighed against the accuracy gain.
- **Boosting's sequential training can't be parallelized the way bagging's can**, making it slower to train, even though inference cost is comparable.
- **Ensembles still need the same reproducibility discipline as any model** in [ML Workflow](ml-workflow.md) — the exact set of trees or boosting rounds needs to be versioned, not just the "type" of ensemble used.

## Common mistakes

- **Assuming an ensemble automatically fixes overfitting regardless of how it's configured** — boosting with too many rounds can still overfit, just later than a single model would.
- **Treating ensemble size as a free accuracy dial** without accounting for the real latency and memory cost of a larger ensemble in production.
- **Losing interpretability without noticing it mattered** — reaching for a large ensemble on a problem where a stakeholder actually needed to understand individual predictions, not just get an accurate one.

## Interview questions

### Basic

- What's the structural difference between bagging and boosting?
- Why does averaging many overfit models tend to reduce overall overfitting?

### Intermediate

- Why can bagging be trained in parallel while boosting generally can't?
- Why might a Random Forest close a bigger accuracy gap than Gradient Boosting on a given dataset, or vice versa?

### Advanced

- A gradient boosting model's test accuracy has started dropping as more boosting rounds are added. What's happening, and how would you fix it?
- You need a model that's both accurate and interpretable for a regulated use case. How would ensemble methods' trade-offs factor into that choice?
