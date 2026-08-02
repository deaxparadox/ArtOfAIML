# Unsupervised Learning

See [Types of Machine Learning](types-of-machine-learning.md) for where unsupervised learning fits among the other categories — this chapter goes deep into what the category actually contains.

## What is it?

Unsupervised learning is the family of techniques where a model finds structure in data with no labels at all — grouping, compressing, or describing data based purely on patterns in the inputs themselves. This chapter catalogs what's actually inside that family: **clustering** and **dimensionality reduction**, and the concrete algorithms for each.

[Types of Machine Learning](types-of-machine-learning.md) framed unsupervised learning as "sorting a pile of photos into groups with no captions to guide you." This chapter is the catalog of actual sorting methods — how you decide what counts as a group, and how you reduce an overwhelming number of measurements down to the few that actually matter.

## Why does it exist?

[Types of Machine Learning](types-of-machine-learning.md) established *why* unsupervised learning gets chosen — no labels exist, or labels aren't even the right kind of output for the task — but treated it as one category. It actually splits into genuinely different problems: clustering answers "which things are similar to each other," dimensionality reduction answers "which of these many measured features actually matter, and can the rest be compressed away." This chapter exists to name that split and the concrete algorithms within it.

**Clustering vs. dimensionality reduction, as a real decision:** reach for clustering when the goal is grouping similar items — customer segments, topic clusters. Reach for dimensionality reduction when there are many correlated or redundant features and the goal is a smaller set that captures most of the actual signal — for visualization, for speeding up a downstream model, or for reducing noise, exactly as [Embeddings](../embeddings/embeddings.md)' TF-IDF + SVD example already did. These solve different problems; picking the wrong one doesn't just underperform, it doesn't even answer the question being asked.

**Within clustering itself, a further decision:** [Types of Machine Learning](types-of-machine-learning.md) already used `KMeans`, which assumes clusters are roughly round and similarly sized, and requires choosing the number of clusters in advance. **DBSCAN** groups by density instead, finding arbitrarily-shaped clusters and automatically flagging outliers as noise — at the cost of tuning a different, less intuitive pair of parameters (a radius and a minimum point count, rather than "number of clusters").

## How does it work?

**Clustering** assigns each point to a group based on similarity, with no label ever specifying a "correct" grouping. **Dimensionality reduction** finds a smaller set of new dimensions — often linear combinations of the original features, as in [PCA](pca.md) or SVD — that preserve most of the original data's variance or structure.

**`KMeans` fits via Lloyd's algorithm**, an iterative two-step loop with no gradient descent involved at all:

1. Pick `k` initial centroids (often `k` random data points).
2. **Assign step**: assign every point to its nearest centroid.
3. **Update step**: move each centroid to the mean of the points now assigned to it.
4. Repeat steps 2–3 until assignments stop changing — a fixed point, since centroids that don't move produce the same assignments next round, and vice versa.

**DBSCAN** works by density instead of iterative refinement: for each point, it counts how many other points fall within a radius `eps`; a point with at least `min_samples` neighbors within that radius is a **core point**, points reachable from a core point join its cluster, and any point reachable from no core point is labeled noise — which is also how DBSCAN detects outliers automatically, something Lloyd's algorithm has no mechanism for at all.

## Example

**Mechanism verification: does a from-scratch Lloyd's algorithm converge to the same centroids `KMeans` finds?**

```python
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.8, random_state=0)

rng = np.random.default_rng(0)
init_idx = rng.choice(len(X), size=3, replace=False)
centroids = X[init_idx].copy()

for iteration in range(300):
    distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    assignments = np.argmin(distances, axis=1)
    new_centroids = np.array([X[assignments == k].mean(axis=0) for k in range(3)])
    shift = np.linalg.norm(new_centroids - centroids)
    centroids = new_centroids
    if shift < 1e-6:
        print(f"converged at iteration {iteration}")
        break

sk = KMeans(n_clusters=3, init=X[init_idx], n_init=1, random_state=0).fit(X)
print("manual centroids:\n", centroids)
print("sklearn centroids:\n", sk.cluster_centers_)
```

```text
converged at iteration 11
manual centroids:
 [[-1.75331622  2.9376426 ]
 [ 0.84069413  4.35679782]
 [ 2.05130138  1.06134965]]
sklearn centroids:
 [[-1.75331622  2.9376426 ]
 [ 0.84069413  4.35679782]
 [ 2.05130138  1.06134965]]
```

The manual assign-update loop matches `KMeans`'s own centroids exactly, given the same starting points. The centroid shift between iterations doesn't shrink monotonically along the way — it actually grows for several iterations (some points keep flipping between two competing clusters) before finally collapsing to zero at iteration 11. That's a real, honest picture of Lloyd's algorithm: it's guaranteed to eventually stop, not guaranteed to improve smoothly on every single step.

`KMeans` genuinely fails on data where the true clusters aren't round — two concentric rings:

```python
from sklearn.datasets import make_circles
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import adjusted_rand_score

X, y_true = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=0)

kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
dbscan = DBSCAN(eps=0.2, min_samples=5).fit(X)

print("KMeans ARI:", adjusted_rand_score(y_true, kmeans.labels_))
print("DBSCAN ARI:", adjusted_rand_score(y_true, dbscan.labels_))
```

```text
KMeans ARI: -0.003
DBSCAN ARI: 1.000
```

The Adjusted Rand Index measures agreement with the true ring structure (1.0 is perfect, 0 is random). `KMeans` scores essentially at random — it's mathematically incapable of separating an inner ring from an outer one, since both rings are centered on the same point and no round boundary can divide them. `DBSCAN` recovers the true rings exactly, because it groups by local density rather than assuming any particular shape. Getting there required tuning `eps` — the same "less intuitive parameters" trade-off named above: at `eps=0.15`, DBSCAN fragmented the rings into nine pieces instead of two; only `eps=0.2` recovered the clean two-cluster structure.

## Where is it used?

Customer segmentation, anomaly detection, topic discovery, and — via dimensionality reduction — feature compression and visualization ahead of a downstream model, or exploratory analysis of high-dimensional data.

## Advantages

- **Finds structure with no labeling cost at all**, unlike every supervised technique covered in the previous chapter.
- **DBSCAN specifically can discover clusters of any shape** and flags outliers automatically, neither of which `KMeans` can do.
- **Dimensionality reduction reveals which features actually carry signal**, often making downstream analysis or modeling faster and less noisy.

## Limitations

- **`KMeans`'s round-cluster assumption fails outright on non-convex structure**, as the example shows directly — not a minor accuracy hit, a complete failure to find the real groups.
- **DBSCAN's parameters are genuinely harder to reason about than "number of clusters."** A radius and a minimum point count depend on the data's density and scale in a way that isn't always obvious in advance.
- **There's no ground truth to check a clustering result against**, unlike supervised learning's direct correctness signal — a cluster result still needs human judgment to confirm it's actually meaningful, not just numerically distinct.

## Production considerations

- **Clustering results need re-validation as data drifts.** Real-world clusters can shift or merge over time, the same way any statistic can drift — a segmentation that was accurate at launch can quietly stop matching reality.
- **DBSCAN's parameters may need retuning per dataset**, unlike `KMeans`'s single, interpretable `n_clusters` — a parameter set tuned for one dataset's density may fail silently on another with different scale or density.
- **Dimensionality reduction applied before a downstream model needs the same fit-once, apply-consistently discipline** as any other feature transformation covered in [Feature Engineering](feature-engineering.md) — refitting it on new data changes the meaning of the reduced dimensions entirely.

## Common mistakes

- **Defaulting to `KMeans` without checking whether the data's clusters are actually round-ish**, exactly the assumption this chapter's example shows failing outright.
- **Treating DBSCAN's default or first-guess parameters as final**, without checking, as this chapter did, whether they produce a sensible number of clusters at all.
- **Skipping human review of a clustering result**, assuming a numerically distinct grouping is automatically a meaningful one.

## Interview questions

### Basic

- What's the difference between clustering and dimensionality reduction?
- What assumption does `KMeans` make about cluster shape?
- What are the two steps `KMeans` repeats until convergence?

### Intermediate

- Why does `KMeans` fail on concentric-ring-shaped clusters, and why does DBSCAN succeed?
- Why is there no direct way to check a clustering result's "accuracy" the way there is for supervised learning?
- Why isn't `KMeans`'s centroid movement guaranteed to shrink smoothly on every iteration, even though the algorithm is guaranteed to eventually converge?

### Advanced

- A clustering result that was validated at launch has become less meaningful over time, with no code changes. What would you investigate?
- How would you decide between `KMeans` and DBSCAN for a new, unfamiliar dataset, before knowing the true cluster shapes in advance?
