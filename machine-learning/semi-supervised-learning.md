# Semi-Supervised Learning

See [Types of Machine Learning](types-of-machine-learning.md) for where semi-supervised learning fits among the other categories — this chapter goes deep into how it actually works.

## What is it?

Semi-supervised learning trains on a mix of a small amount of labeled data and a large amount of unlabeled data, using the unlabeled data to improve on what the small labeled set alone could teach. This chapter is about the actual mechanics of how the unlabeled majority contributes, not just that it's "used somehow."

[Types of Machine Learning](types-of-machine-learning.md) framed this as "most of the training data is unlabeled, with only a small labeled subset." This chapter covers the specific techniques that let that unlabeled majority actually participate, rather than sitting unused.

## Why does it exist?

[Types of Machine Learning](types-of-machine-learning.md) already established the motivation: labeling is expensive, but unlabeled data is often abundant and nearly free. This chapter goes one level deeper — how does a model extract value from data with no label attached? Naive intuition says you can't learn anything from data with no answer key. Semi-supervised learning exists to show that's false: unlabeled data still carries information about the structure of the input space — which points sit near which — and that structural information can be combined with a small labeled set to build a better model than the labeled set alone could support.

**When semi-supervised learning is worth the complexity:** if the labeled set alone is too small to train a good model, and there's a meaningful budget gap between "label a little" and "label everything," semi-supervised techniques earn their complexity. If the small labeled set already performs well enough on its own, the added pipeline complexity isn't worth it. If labeling budget is genuinely unconstrained, fully supervised learning on a fully labeled set is simpler and usually performs at least as well.

## How does it work?

**Self-training** trains a model on the small labeled set, uses it to predict labels for the unlabeled data, treats the most confident predictions as if they were true labels, and retrains on the combined set — a simple, iterative bootstrap. **Label propagation / label spreading** assumes nearby points, measured by similarity — a direct application of [Similarity](../embeddings/similarity.md) — likely share a label, and spreads known labels outward to nearby unlabeled points through the data's underlying structure.

## Example

With only 8 labeled points out of 300 training examples — genuinely scarce labels — comparing a supervised model trained on just those 8 against label spreading using all 300 points (292 of them unlabeled):

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=400, noise=0.15, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

rng = np.random.RandomState(0)
labeled_idx = rng.choice(len(X_train), 8, replace=False)

# supervised, using only the 8 labeled points
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train[labeled_idx], y_train[labeled_idx])
print(accuracy_score(y_test, knn.predict(X_test)))

# semi-supervised: same 8 labels + all 292 unlabeled points
y_semi = np.full(len(X_train), -1)
y_semi[labeled_idx] = y_train[labeled_idx]
label_spread = LabelSpreading(kernel="knn", n_neighbors=7)
label_spread.fit(X_train, y_semi)
print(accuracy_score(y_test, label_spread.predict(X_test)))
```

```text
supervised (8 labels only): 0.470
semi-supervised (label spreading): 0.750
```

With only 8 labeled examples, the supervised-only model barely beats a coin flip on this two-class problem. Label spreading, using the exact same 8 labels plus the 292 unlabeled points' structure, reaches 75% — the unlabeled data alone didn't say what any individual point's label was, but it revealed which points cluster together, and that structure let the 8 known labels spread outward to their neighbors. The advantage is largest exactly where labels are scarcest; as more labels become available, a purely supervised model on the labeled set catches up and the gap narrows.

## Where is it used?

Any domain where labeling is genuinely expensive relative to data volume — medical imaging (a radiologist's time is scarce, raw scans aren't), fraud detection with a small set of confirmed cases among a huge stream of unlabeled transactions, and speech or text tasks with abundant raw data but limited annotated examples.

## Advantages

- **Extracts real value from unlabeled data**, as the example shows directly, rather than leaving it unused.
- **Reduces labeling cost** for a given target accuracy, compared to needing enough labels for fully supervised learning alone.
- **The benefit is largest exactly when it's needed most** — when labels are scarcest, which is also when it's hardest to justify labeling more data.

## Limitations

- **The benefit isn't guaranteed at every labeled-set size.** As more labels become available, the gap between semi-supervised and purely supervised approaches narrows, and can even briefly favor the simpler supervised model.
- **Depends on the unlabeled data's structure actually being informative.** If nearby points in the input space don't actually share a label in reality, spreading labels through that structure spreads them incorrectly.
- **More complex to implement and tune** than plain supervised learning — an extra kernel and neighbor-count choice, on top of whatever the downstream model itself needs.

## Production considerations

- **The assumption that "nearby points share a label" needs to hold for the specific domain**, not just be assumed by default — verifying it on held-out labeled data before trusting the technique at scale is worth the effort.
- **Self-training can reinforce its own mistakes.** A wrong confident prediction, once treated as a true label and retrained on, can push the model further in the wrong direction rather than correcting it.
- **Retraining as new labels or unlabeled data arrive needs the same reproducibility discipline** as any other pipeline in [ML Workflow](ml-workflow.md) — the exact labeled/unlabeled split used matters for reproducing a given model version.

## Common mistakes

- **Expecting semi-supervised learning to help regardless of how many labels are already available** — the actual example above shows the advantage shrinking, not growing, as labels become less scarce.
- **Assuming the unlabeled data's structure is automatically informative** for the task at hand, without checking that assumption first.
- **Not monitoring self-training for reinforced errors**, letting an early wrong confident prediction compound across retraining iterations.

## Interview questions

### Basic

- What data does semi-supervised learning use that purely supervised learning doesn't?
- What is label spreading, at a high level?

### Intermediate

- Why does semi-supervised learning's advantage shrink as more labeled data becomes available?
- What real-world assumption does label propagation depend on, and what happens if it's wrong?

### Advanced

- Why can self-training reinforce its own mistakes, and how would you detect that happening?
- You have a labeling budget that could either label more examples or none. How would you decide whether semi-supervised learning is worth adopting instead of just labeling more?
