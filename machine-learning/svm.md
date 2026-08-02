# SVM

See [Supervised Learning](supervised-learning.md) for the linear-vs-curved decision boundary trade-off this chapter offers a third option for.

## What is it?

A Support Vector Machine (SVM) finds the decision boundary that maximizes the margin — the distance to the closest point of either class — rather than just any boundary that happens to separate the classes. Only those closest points, the **support vectors**, determine where the boundary sits; every other point could move around freely without changing it at all. Combined with the **kernel trick**, SVMs can also find curved decision boundaries without ever explicitly transforming the data into a higher-dimensional space.

## Why does it exist?

[Supervised Learning](supervised-learning.md#example) already showed `LogisticRegression` losing to `KNeighborsClassifier` and `DecisionTreeClassifier` specifically because its decision boundary is restricted to a straight line. SVM offers a genuinely different answer to that same limitation: keep a single, smooth, mathematically well-defined boundary — unlike a decision tree's jagged, axis-aligned splits — but let that boundary curve through the kernel trick, instead of being stuck as a straight line the way plain Logistic Regression is.

**Linear kernel vs. RBF kernel — the same "does the boundary shape match the data" decision as before, in a new form.** A linear kernel keeps SVM fast, interpretable (each feature still gets a coefficient), and appropriate when the true boundary really is roughly straight, or close enough after feature engineering. An RBF (radial basis function) kernel lets the boundary curve arbitrarily, at the cost of losing that direct interpretability and adding a new hyperparameter (`gamma`, controlling how far each point's influence reaches) that needs tuning.

## How does it work?

1. Among every boundary that would separate the two classes, SVM specifically picks the one maximizing the margin to the nearest point of either class — not just any valid separator.
2. Only the points closest to that boundary — the support vectors — actually determine it; the optimization problem depends entirely on them, and moving a point far from the boundary changes nothing about the solution.
3. The **kernel trick** replaces the dot product at the core of that optimization with a kernel function computing what the dot product *would be* if the data were first mapped into a higher-dimensional space — without ever actually performing that mapping. This lets the boundary be non-linear in the original feature space while the underlying math stays a simple linear separation in the (never-materialized) higher-dimensional one.

## Example

**Margin maximization: only the closest points define the boundary.**

```python
import numpy as np
from sklearn.svm import SVC

X = np.array([[1, 1], [2, 1], [1, 2], [5, 5], [6, 5], [5, 6]])
y = np.array([0, 0, 0, 1, 1, 1])

svm = SVC(kernel="linear", C=1000).fit(X, y)
print("support vectors:", X[svm.support_])
print("margin width:", 2 / np.linalg.norm(svm.coef_[0]))
```

```text
support vectors: [[2 1]
                   [1 2]
                   [5 5]]
margin width: 4.9517
```

Of six points, only three are support vectors — `[2, 1]` and `[1, 2]` from class 0, `[5, 5]` from class 1 — exactly the points nearest the other class. The other three points play no role in the solution at all; removing them and refitting would produce the identical boundary.

**Kernel choice: linear vs. RBF on the same curved-boundary data from Supervised Learning.**

```python
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=300, noise=0.25, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

for kernel in ["linear", "rbf"]:
    model = SVC(kernel=kernel).fit(X_train, y_train)
    print(kernel, accuracy_score(y_test, model.predict(X_test)), len(model.support_))
```

```text
linear: 0.856 (78/210 support vectors)
rbf:    0.967 (62/210 support vectors)
```

The linear kernel scores 85.6% — nearly identical to `LogisticRegression`'s own 84.4% on this exact data in [Supervised Learning](supervised-learning.md#example), which makes sense: both are restricted to a straight boundary on data whose true boundary is curved. Switching only the kernel to RBF, with nothing else about the algorithm changed, recovers the same 96.7% territory the tree and KNN reached — and does it with *fewer* support vectors, not more, showing the curved boundary is actually a tighter fit, not a more complex one in terms of how many points it needs.

## Where is it used?

Classification tasks with a clear margin between classes and a moderate amount of data — text classification was a classic SVM use case before deep learning became dominant there — and any setting where a smooth, well-defined boundary and strong generalization from limited data matter more than training-time speed at very large scale.

## Advantages

- **Margin maximization tends to generalize well, even with a comparatively small amount of training data**, since the boundary is chosen specifically to stay as far as possible from either class.
- **The kernel trick lets a fundamentally linear method model non-linear boundaries** without hand-engineering non-linear features first, as the RBF kernel result shows directly.
- **Only the support vectors determine the boundary**, which can make the fitted model itself compact regardless of how large the original training set was.

## Limitations

- **Training scales poorly to very large datasets** — the core optimization problem's cost grows faster than linearly with the number of training examples, unlike a linear model or a tree.
- **No native probability output.** SVM produces a class decision directly; a probability estimate requires an additional post-hoc calibration step (Platt scaling), not something the model produces natively the way Logistic Regression's sigmoid does.
- **Kernel and hyperparameter choice (`gamma`, `C`) genuinely changes performance**, as the linear-vs-RBF gap in the example shows — the wrong choice can leave an SVM underperforming a much simpler model.

## Production considerations

- **Feature scaling matters for the same reason it does for KNN** in [Feature Engineering](feature-engineering.md#example) — the margin and kernel computations both depend on distances between points, which an unscaled feature can dominate.
- **Training cost needs to be weighed against dataset size before committing to SVM** in a pipeline that retrains regularly — a model that trains acceptably once on a snapshot may become a bottleneck as the training set grows.
- **Kernel choice is a real interpretability decision**, not just an accuracy one — a linear kernel keeps per-feature coefficients meaningful, and switching to RBF for a small accuracy gain trades that away.

## Common mistakes

- **Skipping feature scaling before fitting an SVM**, the same silent-failure risk [Feature Engineering](feature-engineering.md) already demonstrated for KNN.
- **Reaching for an RBF kernel by default** without first checking whether a linear kernel already performs adequately, adding tuning complexity for a gain that may not exist.
- **Treating SVM's calibrated probability output as equivalent to a native probabilistic model's**, when it's a fitted approximation layered on top of a boundary-only model.

## Interview questions

### Basic

- What does an SVM try to maximize when choosing a decision boundary?
- What is a support vector, and why do non-support-vector points not affect the fitted model?

### Intermediate

- What problem does the kernel trick actually solve, and why does it avoid explicitly computing a higher-dimensional mapping?
- Why did the linear-kernel SVM in this chapter's example score almost identically to `LogisticRegression` on the same data?

### Advanced

- When would you prefer an SVM over a tree ensemble or logistic regression for a classification task, and when would that preference reverse as the dataset grows?
- Why doesn't SVM produce probability estimates natively, and what are the practical implications of relying on Platt-scaled probabilities from an SVM in a downstream decision system?
