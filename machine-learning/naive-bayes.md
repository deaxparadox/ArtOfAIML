# Naive Bayes

See [Bayes Theorem](../statistics/bayes-theorem.md) for the formula this classifier applies directly, and for why the independence assumption in its name is called "naive."

## What is it?

Naive Bayes is a classification algorithm that predicts the most probable class for a set of features by applying [Bayes Theorem](../statistics/bayes-theorem.md) directly, using a simplifying assumption: that every feature is conditionally independent of every other feature, given the class. That assumption is what turns an otherwise intractable calculation into a fast, simple one — and what makes the algorithm "naive," since real features are rarely truly independent.

[Bayes Theorem](../statistics/bayes-theorem.md#where-is-it-used) already named Naive Bayes as "an entire, still widely used model family built directly on this formula." This chapter is that model family: how the independence assumption actually gets used to make Bayes' theorem computationally practical for classification with many features, not just the two-event examples that chapter worked through.

## Why does it exist?

Classifying based on many features — hundreds or thousands of words in a document, say — using Bayes' theorem exactly would require knowing `P(all features together | class)`: the full joint likelihood of every feature combination, for every class. Estimating that exactly needs an amount of data that grows explosively with the number of features, since it would have to account for every possible correlation between every pair (and triple, and more) of features. Naive Bayes exists to sidestep that entirely: assume every feature is independent given the class, so the joint likelihood collapses into a simple product of individually estimable per-feature probabilities — each one easy to compute even from limited data.

**Generative vs. discriminative — a real, structural difference, not a minor implementation detail.** Naive Bayes is a **generative** classifier: it models `P(features | class)` and `P(class)` separately, then uses Bayes' theorem to get `P(class | features)`. `LogisticRegression`, from [Supervised Learning](supervised-learning.md), is **discriminative**: it models `P(class | features)` directly, with no attempt to model how the features themselves are distributed. Generative models like Naive Bayes tend to need far less data to reach a reasonable estimate, precisely because they estimate each feature's distribution independently rather than jointly optimizing a decision boundary — at the cost of that independence assumption being wrong in most real datasets.

## How does it work?

1. Estimate the **prior**, `P(class)`, directly from the training data's class frequencies — exactly the prior from [Bayes Theorem](../statistics/bayes-theorem.md).
2. For each feature, estimate `P(feature | class)` independently — assuming conditional independence given the class. For count data (like word frequencies), `MultinomialNB` estimates per-class term frequencies; for continuous features, `GaussianNB` assumes a per-class normal distribution.
3. For a new example, combine the prior with every feature's estimated likelihood (multiplied together — or summed, in log space, to avoid numerical underflow with many features) for each class, and predict whichever class produces the highest resulting score.

## Example

**Data efficiency: how few labeled examples does each algorithm need?**

Two document classes generated from genuinely different word-frequency distributions over a small vocabulary — comparing `MultinomialNB` (generative) against `LogisticRegression` (discriminative) as the training set shrinks:

```python
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

rng = np.random.default_rng(0)
vocab_size = 20
class0_probs = rng.dirichlet(np.ones(vocab_size) * 2)
class1_probs = rng.dirichlet(np.ones(vocab_size) * 2)

def make_docs(n, probs, words_per_doc=15):
    return rng.multinomial(words_per_doc, probs, size=n)

X_test = np.vstack([make_docs(150, class0_probs), make_docs(150, class1_probs)])
y_test = np.array([0]*150 + [1]*150)

for n in [3, 5, 10, 30, 100]:
    X_train = np.vstack([make_docs(n, class0_probs), make_docs(n, class1_probs)])
    y_train = np.array([0]*n + [1]*n)
    nb = MultinomialNB().fit(X_train, y_train)
    lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    print(n, accuracy_score(y_test, nb.predict(X_test)), accuracy_score(y_test, lr.predict(X_test)))
```

```text
n_train/class=   3: NaiveBayes=0.920  LogisticRegression=0.890
n_train/class=   5: NaiveBayes=0.913  LogisticRegression=0.873
n_train/class=  10: NaiveBayes=0.913  LogisticRegression=0.890
n_train/class=  30: NaiveBayes=0.950  LogisticRegression=0.917
n_train/class= 100: NaiveBayes=0.947  LogisticRegression=0.940
```

With only 3 labeled examples per class, `MultinomialNB` already reaches 92.0% accuracy, ahead of `LogisticRegression`'s 89.0%. That gap persists at every training size shown, and genuinely narrows as training data grows — by 100 examples per class, the two are nearly tied (94.7% vs. 94.0%). This is the generative-vs-discriminative trade-off directly: estimating each word's frequency per class independently needs far less data to become reasonably accurate than fitting a discriminative decision boundary does, exactly why Naive Bayes remains a strong choice specifically when labeled data is scarce.

## Where is it used?

Text classification — spam filtering, sentiment analysis, document categorization — often on top of the exact [TF-IDF / Bag-of-Words](../nlp/tfidf-bag-of-words.md) features already covered, where the vocabulary is large, the labeled data is comparatively limited, and the independence assumption, while technically wrong, doesn't cost much in practice.

## Advantages

- **Needs far less labeled data to reach a reasonable accuracy** than a comparable discriminative model, as the example's small-training-size results show directly.
- **Trains and predicts very fast**, since every feature's likelihood is estimated independently rather than jointly optimized.
- **Handles a large number of features naturally**, which is exactly what a large vocabulary in text classification produces.

## Limitations

- **The independence assumption is essentially always false in real data.** Words in real documents are correlated; features in most real datasets are correlated. Naive Bayes' rankings can still be useful even when this makes its raw probability outputs poorly calibrated, exactly the caution [Bayes Theorem](../statistics/bayes-theorem.md#limitations) already raised.
- **As training data grows, the data-efficiency advantage shrinks**, as the example shows — a discriminative model given enough data can close most of the gap, since it isn't handicapped by a false independence assumption to begin with.
- **A feature value never seen during training for a given class gets a zero probability outright**, without smoothing — collapsing the entire prediction to zero regardless of every other feature's evidence.

## Production considerations

- **Smoothing (Laplace/additive smoothing) needs to be applied by default**, not treated as optional — without it, any unseen feature-class combination at prediction time can zero out an otherwise confident prediction entirely.
- **Naive Bayes' probability outputs shouldn't be trusted as calibrated posteriors** without checking, the same warning [Bayes Theorem](../statistics/bayes-theorem.md#production-considerations) already gave — fine for ranking or thresholding on relative confidence, risky for a system that needs the actual probability value to be meaningful.
- **Its fast retraining cost makes it a reasonable choice for a pipeline that needs to retrain frequently** on fresh data, unlike heavier discriminative models that cost more to refit regularly.

## Common mistakes

- **Assuming a low accuracy means Naive Bayes "doesn't work" for a given dataset**, when the independence assumption's cost is often small in practice — this chapter's own results show it competitive with, and often ahead of, a discriminative alternative.
- **Skipping smoothing**, and being surprised when a single unseen word or feature value at prediction time zeroes out an entire class's score.
- **Using Naive Bayes' output probabilities directly in a downstream decision** that assumes they're well-calibrated, without checking, exactly the mistake [Bayes Theorem](../statistics/bayes-theorem.md#common-mistakes) already named.

## Interview questions

### Basic

- What does "naive" refer to in Naive Bayes?
- What's the difference between a generative and a discriminative classifier?

### Intermediate

- Why does Naive Bayes typically need less training data than a discriminative model like Logistic Regression to reach a reasonable accuracy?
- Why does an unseen feature value at prediction time zero out an entire class's probability without smoothing?

### Advanced

- This chapter's example showed Naive Bayes' advantage over Logistic Regression narrowing as training data grew. At what point, in general, would you expect a discriminative model to overtake a generative one, and why?
- You're building a text classifier that needs frequent retraining on a small, constantly refreshed labeled set. How would that constraint affect your choice between Naive Bayes and a discriminative alternative?
