# Supervised Learning

See [Types of Machine Learning](types-of-machine-learning.md) for where supervised learning fits among the other categories — this chapter goes deep into what the category actually contains.

## What is it?

Supervised learning is the family of techniques where a model learns from examples that already include the correct answer — a label. This chapter is the catalog of what's actually inside that family: the two problem types it splits into (classification, regression), and the concrete algorithms used to solve them.

[Types of Machine Learning](types-of-machine-learning.md) introduced supervised learning as "studying with an answer key." This chapter is the catalog of study methods available once you have that answer key — different algorithms make different assumptions about how to use the labeled examples to generalize to new ones.

## Why does it exist?

[Types of Machine Learning](types-of-machine-learning.md) established *why* supervised learning gets chosen — labeled data exists, a "correct answer" concept applies — but treated it as one undifferentiated category. In practice, "supervised learning" spans wildly different algorithms with different assumptions, strengths, and costs. A linear model, a tree-based model, and a distance-based model all "do supervised learning," but work, scale, and fail in completely different ways. This chapter exists to give the actual menu of options and the decision criteria for picking among them, rather than treating "use a supervised model" as a single choice.

**Linear vs. tree-based vs. distance-based models, as a real algorithm-selection decision:**

- **Linear models** (`LinearRegression`, `LogisticRegression` — already used throughout this handbook) are fast and interpretable — each feature's contribution is a single coefficient — and work well when the true relationship is roughly linear, or becomes so after feature engineering. They underperform when the real relationship has non-linear structure they structurally can't represent.
- **Tree-based models** (decision trees, and their ensembles, covered in the next chapter) naturally capture non-linear relationships and feature interactions without manual feature engineering, and don't need feature scaling — the same scale-invariance already noted in [Feature Engineering](feature-engineering.md#common-mistakes). A single tree can overfit easily on its own, and becomes less interpretable once ensembled.
- **Distance-based models** (k-nearest neighbors) make no assumption about the relationship's shape at all, but require feature scaling — exactly the [Feature Engineering](feature-engineering.md#example) example already showed — and get slower and more memory-hungry as the dataset grows, since prediction means comparing against many stored training points.

## How does it work?

**Classification** predicts a discrete category; **regression** predicts a continuous number — the same distinction [Types of Machine Learning](types-of-machine-learning.md) already drew, with different loss functions and evaluation metrics for each. Training fits a model's parameters by minimizing a loss function measuring how wrong its predictions are on the training data; prediction applies those learned parameters to new, unseen input.

## Example

The same classification task, three different algorithms, on data that isn't linearly separable — points arranged in two interleaving crescents:

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=300, noise=0.25, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

models = {
    "LogisticRegression": LogisticRegression(),
    "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=5, random_state=0),
    "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=5),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(name, accuracy_score(y_test, model.predict(X_test)))
```

```text
LogisticRegression: 0.844
DecisionTreeClassifier: 0.933
KNeighborsClassifier: 0.967
```

Same data, same split, three genuinely different results. `LogisticRegression` draws a single straight decision boundary, and this data's true boundary is curved — it structurally cannot fit it as well as the other two. `KNeighborsClassifier` wins here because the crescents' local neighborhoods are informative: nearby points really do share a class. This is exactly the kind of gap the "decision" section above is about — the algorithm isn't a minor implementation detail, it's the difference between 84% and 97% accuracy on the same problem.

## Where is it used?

Every classification and regression task already shown in this handbook — spam detection, price prediction, medical diagnosis, credit scoring — anywhere labeled historical data exists and the goal is to predict a label or number for new, unlabeled input.

## Advantages

- **A direct, measurable per-example correctness signal**, since every training example has a known right answer, making a supervised model easy to evaluate objectively.
- **A wide, mature menu of algorithms** with well-understood, well-documented trade-offs, built up over decades of use.
- **Directly optimizable** — a clear loss function to minimize is exactly what makes training a supervised model a well-defined problem in the first place.

## Limitations

- **Entirely dependent on labeled data**, which is often expensive or slow to produce — the same labeling-cost trade-off [Types of Machine Learning](types-of-machine-learning.md) already raised for semi-supervised learning.
- **No single algorithm dominates every problem.** The example above shows a linear model losing by 12 points of accuracy on data it was structurally unsuited for — the "right" algorithm depends on the data's actual shape.
- **Learns only what the labels reward.** A supervised model optimizes exactly the metric it's trained against, and nothing else — a subtly wrong label or a poorly chosen target metric shapes the model just as strongly as a correct one would.

## Production considerations

- **Algorithm choice affects more than accuracy** — training time, inference latency, memory footprint, and interpretability all vary by algorithm family, and the best-scoring model in an offline test isn't automatically the best production choice if it's too slow or too opaque for the use case.
- **A model's assumptions need to match the data it will actually see in production**, not just the data it was validated on — a linear model that performed acceptably on a validation set can fail badly the moment real-world data has more non-linear structure than that set happened to capture.
- **Model choice interacts directly with [Feature Engineering](feature-engineering.md)'s own trade-offs** — scaling matters for KNN and linear models but not for trees, which changes the entire preprocessing pipeline depending on which algorithm family is chosen.

## Common mistakes

- **Defaulting to one familiar algorithm for every problem**, without checking whether the data's actual structure fits its assumptions — exactly what the linear model's underperformance in the example demonstrates.
- **Comparing algorithms on accuracy alone**, ignoring training time, inference cost, and interpretability differences that matter just as much in production.
- **Assuming a more complex algorithm (like KNN or a deep tree) is always better**, without checking whether that complexity is actually warranted by the data, or is just adding overfitting risk for no real gain.

## Interview questions

### Basic

- What are the two main problem types within supervised learning?
- Why does k-nearest neighbors need feature scaling when a decision tree doesn't?

### Intermediate

- Why might a linear model perform noticeably worse than a tree-based or distance-based model on the same dataset?
- What does it mean for an algorithm's assumptions to "match" or "not match" the data's true structure?

### Advanced

- You have a supervised learning problem with abundant, well-labeled data and a hard latency requirement in production. How would that constraint change which algorithm family you'd choose, even if a slower one scored slightly higher offline?
- Why doesn't the best-performing algorithm in an offline evaluation automatically make it the right production choice?
