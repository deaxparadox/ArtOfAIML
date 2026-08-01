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

**Clustering** assigns each point to a group based on similarity, with no label ever specifying a "correct" grouping. **Dimensionality reduction** finds a smaller set of new dimensions — often linear combinations of the original features, as in PCA/SVD — that preserve most of the original data's variance or structure.

## Example

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

### Intermediate

- Why does `KMeans` fail on concentric-ring-shaped clusters, and why does DBSCAN succeed?
- Why is there no direct way to check a clustering result's "accuracy" the way there is for supervised learning?

### Advanced

- A clustering result that was validated at launch has become less meaningful over time, with no code changes. What would you investigate?
- How would you decide between `KMeans` and DBSCAN for a new, unfamiliar dataset, before knowing the true cluster shapes in advance?
