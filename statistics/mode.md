# Mode

See [Mean](mean.md) and [Median](median.md) for the other two measures of central tendency this chapter completes.

## What is it?

The mode is the most frequently occurring value in a dataset. Unlike the mean or median, it isn't restricted to numbers — it's just a count of which value shows up most, so it works on categories just as well as on numbers.

Where the mean is a balance point and the median is a middle-of-the-line position, the mode is neither — it's a popularity count. It doesn't care how large a value is or where it ranks, only how often it appears.

## Why does it exist?

The mean and median both require data that can be added or ordered — you can't average or sort `"red", "blue", "green"`. The mode exists because it's the only one of the three central-tendency measures that works on purely categorical data: it just counts occurrences and picks the winner.

**When to reach for the mode instead of mean or median:** whenever the data is categorical (most common browser, most common error code, most common product purchased), or whenever you specifically want "what's most common" rather than "what's typical in a numeric sense." **When mean or median is still the better choice:** for numeric data where magnitude or rank actually carries meaning — the mode of a set of exact response times, for instance, is usually meaningless, since continuous values rarely repeat at all.

## How does it work?

Count the occurrences of each distinct value. Whichever value (or values, if there's a tie) has the highest count is the mode. A dataset can have one mode, multiple tied modes ("multimodal"), or no meaningful mode at all if every value is equally rare.

## Example

Counting HTTP status codes from a slice of server logs:

```python
from collections import Counter

status_codes = [200, 200, 200, 404, 500, 200, 404]
print(Counter(status_codes).most_common())  # -> [(200, 4), (404, 2), (500, 1)]
```

`200` is the mode — it's the most common outcome, nothing more, nothing less.

The more useful version of this idea in machine learning is the **majority-class baseline**. [Mean](mean.md) introduced "always predict the average" as the standard baseline a regression model has to beat; for classification, the equivalent is "always predict the most common class" — the mode of the target variable.

```python
from collections import Counter

y = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0]  # 18 zeros, 2 ones
counts = Counter(y)
majority_class, majority_count = counts.most_common(1)[0]
baseline_accuracy = majority_count / len(y)
print(majority_class, baseline_accuracy)  # -> 0 0.9
```

Just predicting the mode for every example gets 90% accuracy here, for free, with no model at all. That number matters directly for interpreting a real model's performance — see Common mistakes below.

## Where is it used?

- **Majority-class baseline** for classification problems, the direct counterpart to the mean-based baseline for regression.
- **Missing value imputation for categorical features** — scikit-learn's `SimpleImputer(strategy="most_frequent")`, the categorical equivalent of mean or median imputation from [Feature Engineering](../machine-learning/feature-engineering.md).
- **Frequency analysis** — most common error code, most common browser or device, most common category in a log or dataset.
- **Detecting multimodal patterns** — a dataset with two clearly separate modes is often a sign of two distinct underlying groups, which the mean or median would blend together into one misleading number.

## Advantages

- **The only central-tendency measure that works on non-numeric data** — categories have no meaningful average or median.
- **Simple and intuitive** — "what shows up most" needs no explanation.
- **Surfaces multimodal structure** that a single mean or median would otherwise hide entirely.

## Limitations

- **Often meaningless on continuous numeric data.** Exact, high-precision values (response times to the millisecond) rarely repeat, so there may be no meaningful mode at all unless the data is binned first.
- **Ignores magnitude entirely** — more than the median, which at least uses rank. The mode doesn't distinguish between values that are close together and values that are far apart, only which one repeats most.
- **A single mode hides multimodal data.** If a dataset actually has two distinct clusters, reporting one mode (or, worse, the mean or median) obscures that there were ever two groups to begin with.

## Production considerations

- **The majority-class baseline exposes class imbalance immediately.** If just predicting the mode already gets 90% accuracy, then a model reporting "95% accuracy" is only 5 points better than doing nothing — the exact reasoning behind why a model with very high raw accuracy can still be a bad model, a question already raised in [What is Machine Learning](../machine-learning/what-is-machine-learning.md#interview-questions) and [ML Workflow](../machine-learning/ml-workflow.md#interview-questions).
- **Finding the exact mode over a large, high-cardinality stream is expensive** — the most common URL path across millions of log lines is a "heavy hitters" problem, and production systems typically use approximate frequency-counting structures instead of tracking exact counts for every distinct value.
- **A shifting mode is itself an operational signal.** If the most common HTTP status code moves from 200 to 500 over time, that shift matters more than the raw mode value at any single point in time.

## Common mistakes

- **Using the mode as a summary for continuous numeric data without binning first.** On raw continuous values, there's often no repeated value at all, making the "mode" arbitrary or undefined.
- **Not checking the majority-class baseline before evaluating a classifier's accuracy.** A high accuracy number means very little without knowing what a trivial mode-only prediction would already have scored.
- **Reporting a single mode for a multimodal dataset**, hiding that the data actually represents two or more distinct groups worth investigating separately rather than summarizing with one number.

## Interview questions

### Basic

- What is the mode, and how is it different from the mean and median?
- Why can the mode be used on categorical data when the mean can't?

### Intermediate

- What is the majority-class baseline in a classification problem, and how does it relate to the mode?
- Why is the mode often a poor summary statistic for continuous numeric data?

### Advanced

- A classification model reports 95% accuracy. How does knowing the mode of the target variable change how you'd interpret that number?
- How would you find the most frequent value(s) in a very large, high-cardinality data stream, where exact counting isn't feasible?
