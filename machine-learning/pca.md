# PCA

See [Unsupervised Learning](unsupervised-learning.md) for where dimensionality reduction fits among unsupervised techniques, and [Embeddings](../embeddings/embeddings.md) for SVD, a closely related dimensionality reduction technique already used directly on text data.

## What is it?

PCA (Principal Component Analysis) finds a new set of axes — **principal components** — that are linear combinations of the original features, ordered so the first captures the most variance in the data, the second captures the most of whatever variance remains, and so on, with every component constrained to be uncorrelated with the ones before it. Keeping only the first few components compresses the data into fewer dimensions while retaining as much of its original variance as mathematically possible for that number of dimensions.

## Why does it exist?

[Unsupervised Learning](unsupervised-learning.md#why-does-it-exist) already named dimensionality reduction as the answer when many features are correlated or redundant and the goal is a smaller set that captures most of the actual signal. PCA is the classical, most direct way of doing that reduction — it derives the reduced dimensions straight from the data's own covariance structure, rather than requiring a choice of which original features to keep or drop by hand.

**PCA vs. manually selecting features — a real, structural difference.** Manually dropping features, as touched on in [Feature Engineering](feature-engineering.md), discards whatever information those specific features carried. PCA doesn't drop original features at all — it creates new ones that blend them, capturing shared signal across correlated originals that outright dropping would lose. The cost: a principal component is a mix of many original features at once, not any single recognizable one, trading interpretability for a more information-preserving compression.

## How does it work?

1. **Center** the data (subtract the mean of each feature), so the variance calculation isn't skewed by an offset.
2. Compute the **covariance matrix** — how every pair of features varies together.
3. Find the covariance matrix's **eigenvectors and eigenvalues**: each eigenvector gives a direction (a principal component), and its corresponding eigenvalue gives how much variance lies along that direction.
4. Sort components by eigenvalue, descending, and keep the top `k` — the eigenvalues directly give the **explained variance**, and dividing each by their total sum gives the **explained variance ratio** for that component.
5. **Project** the original centered data onto the kept eigenvectors to get the final, reduced-dimension representation.

## Example

**Mechanism verification: does a from-scratch eigendecomposition match `sklearn.decomposition.PCA` exactly?**

Three features — two strongly correlated, one independent noise:

```python
import numpy as np
from sklearn.decomposition import PCA

rng = np.random.default_rng(0)
x1 = rng.normal(size=300)
x2 = 2 * x1 + rng.normal(scale=0.3, size=300)  # strongly correlated with x1
x3 = rng.normal(size=300)                       # independent noise
X = np.column_stack([x1, x2, x3])
X_centered = X - X.mean(axis=0)

cov = np.cov(X_centered, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov)
order = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]

print("manual explained variance ratio:", eigvals / eigvals.sum())

pca = PCA(n_components=2).fit(X)
print("sklearn explained variance ratio:", pca.explained_variance_ratio_)
```

```text
manual explained variance ratio:  [0.8588 0.1385 0.0027]
sklearn explained variance ratio: [0.8588 0.1385]
```

The manual eigendecomposition's ratios match `PCA`'s to four decimal places — component 0 alone captures 85.9% of the total variance, entirely from the strong `x1`/`x2` correlation, and component 1 picks up most of the rest. The independent noise feature, `x3`, ends up almost entirely in the discarded third component (0.27% of total variance) — confirming PCA correctly separated genuine shared signal from unrelated noise, using nothing but the data's own covariance structure. (The eigenvectors themselves matched too, once accounted for: an eigenvector's sign is mathematically arbitrary, so `sklearn`'s components came out sign-flipped relative to the manual ones on one axis — a real, expected quirk, not a discrepancy.)

## Where is it used?

Feature compression ahead of a downstream model, visualization of high-dimensional data by reducing it to 2–3 dimensions for plotting, noise reduction, and the same family of technique [Embeddings](../embeddings/embeddings.md#example) already used directly via SVD to compress TF-IDF vectors.

## Advantages

- **Combines correlated features into fewer, uncorrelated ones**, capturing shared signal that dropping features outright would lose entirely.
- **A deterministic solution derived directly from the data's own covariance structure**, with no hyperparameter to tune besides how many components to keep.
- **Effectively separates genuine shared signal from noise**, as the example's independent noise feature — correctly relegated to the smallest component — shows directly.

## Limitations

- **Components are linear combinations of many original features**, losing the direct interpretability a single named feature has — a real cost when interpretability matters as much as compression.
- **Only captures linear relationships between features.** PCA has no way to discover the kind of curved, non-convex structure [DBSCAN](unsupervised-learning.md#example) handles for clustering.
- **Sensitive to feature scale** — a feature on a much larger raw scale can dominate the variance calculation for reasons having nothing to do with actual signal, unless features are standardized first.

## Production considerations

- **PCA needs the same fit-once, apply-consistently discipline** [Unsupervised Learning](unsupervised-learning.md#production-considerations) already flagged for any dimensionality reduction — refitting on new data changes what every component means entirely, breaking comparability with anything computed under the old fit.
- **How many components to keep is a real trade-off**, usually made by picking enough to reach a target cumulative explained variance (95% is a common default) rather than an arbitrary fixed count.
- **Feature scaling should happen before PCA**, for the same reason it matters for KNN in [Feature Engineering](feature-engineering.md#example) — otherwise the "top" component may just reflect whichever feature happens to have the largest raw scale.

## Common mistakes

- **Applying PCA without scaling features first**, letting a large-scale feature dominate the top component for reasons unrelated to actual signal.
- **Treating a principal component as directly interpretable** ("component 1 is basically age") without checking — it's a blend of every original feature, weighted by the eigenvector, not a stand-in for any one of them.
- **Refitting PCA separately on training and production data**, silently changing what each component represents between the two and breaking any comparison built on top of it.

## Interview questions

### Basic

- What does a principal component represent?
- What does "explained variance ratio" tell you about a component?

### Intermediate

- Why does PCA require centering the data before computing the covariance matrix?
- Why does feature scaling matter for PCA in the same way it matters for KNN?

### Advanced

- You reduce a dataset to two principal components capturing 95% of total variance, and a downstream model's accuracy drops noticeably compared to using all original features. What would you investigate?
- How would you decide how many components to keep for a specific downstream task, versus defaulting to a fixed rule like "95% cumulative variance"?
