# Regularization

See [Bias vs Variance](bias-vs-variance.md) for the diagnosis this chapter provides a concrete fix for.

## What is it?

Regularization is a technique for reducing a model's variance by penalizing complexity directly in its training objective — adding a term to the loss function that discourages large or numerous coefficients, so the model is pushed toward a simpler solution even if a more complex one would fit the training data slightly better.

[Bias vs Variance](bias-vs-variance.md) showed a degree-15 polynomial overfitting catastrophically, because it had more free parameters than the data could meaningfully constrain. Regularization is a direct, deliberate constraint on those parameters: instead of trusting the model to naturally stay simple, you explicitly penalize it for becoming too complex — the way a strict editor penalizes an overly long sentence even if every word in it is individually correct.

## Why does it exist?

[Bias vs Variance](bias-vs-variance.md) established the diagnosis — a model with more capacity than the data can support tends toward high variance — but its own fixes were "more capacity" (for bias) or "more data, or a simpler model" (for variance). Regularization exists as a third lever, specifically for variance, that requires neither collecting more data nor manually picking a simpler model architecture. The model keeps its full flexibility; what changes is how much it's allowed to rely on any single feature, enforced by adding a complexity penalty directly into what the model optimizes.

**L1 (Lasso) vs. L2 (Ridge), as a real, concrete choice.** L2 penalizes the sum of squared coefficients, shrinking all of them toward zero but rarely to exactly zero — good when most features are believed to contribute at least a little, and the goal is to keep them all, just dampened. L1 penalizes the sum of absolute coefficient values, which can shrink some coefficients to exactly zero — effectively performing automatic feature selection, useful when many features are suspected to be irrelevant and the model itself should identify and drop them.

**The regularization strength itself is a dial, not a free win.** Too little regularization doesn't meaningfully constrain the model, and it still overfits. Too much pushes the model toward underfitting, trading variance for bias it didn't need to take on. Regularization strength is the bias-variance trade-off, made into a single tunable number.

## How does it work?

A standard loss function minimizes prediction error alone — squared error, as used throughout this handbook. A regularized loss minimizes prediction error *plus* a penalty term proportional to the model's coefficient magnitudes, scaled by a strength parameter (commonly called `alpha`). A higher `alpha` means a stronger penalty, a simpler model, lower variance, and higher bias; a lower `alpha` means the opposite — this is the bias-variance dial from the previous chapter, made directly controllable.

## Example

The same degree-15 polynomial overfitting setup from [Bias vs Variance](bias-vs-variance.md#example), now regularized with Ridge's closed-form solution on the raw polynomial basis:

```python
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
X = np.linspace(0, 10, 30)
y = np.sin(X) + rng.normal(0, 0.2, size=X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

degree = 15
V_train = np.vander(X_train, degree + 1, increasing=True)
V_test = np.vander(X_test, degree + 1, increasing=True)

def ridge_fit(V, y, alpha):
    n_features = V.shape[1]
    return np.linalg.solve(V.T @ V + alpha * np.eye(n_features), V.T @ y)

for alpha in [0.0, 0.0001, 0.01]:
    coeffs = ridge_fit(V_train, y_train, alpha)
    test_mse = mean_squared_error(y_test, V_test @ coeffs)
    print(f"alpha={alpha}: test_mse={test_mse:.4f}")
```

```text
alpha=0.0:    test_mse=200698.0118
alpha=0.0001: test_mse=126.2321
alpha=0.01:   test_mse=0.1814
```

With no regularization at all, the same catastrophic overfitting from [Bias vs Variance](bias-vs-variance.md) shows up again — an unregularized degree-15 fit is numerically unstable, and solving it via the normal equations directly (rather than `np.polyfit`'s more numerically careful method) makes it even more extreme here, underscoring that "how catastrophic" an ill-conditioned fit looks can depend on exactly which linear algebra routine solves it, not just the model itself. A tiny regularization strength, `alpha=0.0001`, already cuts the test error by three orders of magnitude. `alpha=0.01` brings it down to 0.18 — the same model family, same degree-15 flexibility, with nothing changed except a complexity penalty added to what it optimizes.

## Where is it used?

Any linear or generalized linear model prone to overfitting with many features or high-degree polynomial terms, and any situation, per L1's feature-selection property, where identifying which of many features actually matter is itself part of the goal.

## Advantages

- **Reduces variance without needing more data or a different model architecture** — the same flexible model, constrained rather than replaced.
- **L1 regularization performs automatic feature selection**, useful when many available features are suspected to be irrelevant.
- **A single tunable parameter (`alpha`) controls the entire bias-variance trade-off**, rather than requiring a fundamentally different model choice.

## Limitations

- **Regularization strength has no universally correct value.** It has to be tuned for the specific dataset — too little changes nothing, too much introduces the exact underfitting problem [Bias vs Variance](bias-vs-variance.md) already warned about.
- **L1 and L2 make different, real assumptions about the true underlying relationship** — choosing the wrong one doesn't just underperform slightly, it can actively fight against the actual structure in the data (e.g., L1 zeroing out a feature that genuinely matters).
- **Doesn't fix a fundamentally wrong model choice.** Regularizing a linear model doesn't help if the true relationship is non-linear in a way no linear model, however constrained, can represent.

## Production considerations

- **`alpha` needs to be selected using held-out validation data, not guessed**, the same evaluation discipline [ML Workflow](ml-workflow.md) already established for any model choice.
- **Regularization strength interacts with feature scale**, since a penalty on coefficient size only makes sense when features are on comparable scales — the same [Feature Engineering](feature-engineering.md) scaling discussion applies directly here.
- **A regularized model's coefficients are systematically biased toward zero by design** — worth remembering if the coefficients themselves are being interpreted directly, not just used for prediction.

## Common mistakes

- **Applying regularization without scaling features first**, letting the penalty fall disproportionately on whichever features happen to have larger raw magnitudes, regardless of their actual importance.
- **Picking `alpha` once and never revisiting it**, when the right strength depends on the data and can change as the data itself changes over time.
- **Assuming regularization is a substitute for getting the underlying model choice right**, rather than a refinement on top of an already reasonable model.

## Interview questions

### Basic

- What does regularization add to a model's training objective?
- What's the practical difference between L1 and L2 regularization?

### Intermediate

- Why does increasing regularization strength move a model from high variance toward high bias?
- Why does L1 regularization perform feature selection while L2 doesn't?

### Advanced

- Why did solving the degree-15 polynomial fit via the normal equations directly produce an even more extreme unregularized error than `np.polyfit` did in Bias vs Variance? What does that imply about "the" unregularized baseline for an ill-conditioned problem?
- How would you choose between L1 and L2 regularization for a dataset with hundreds of features, many suspected to be irrelevant?
