# Canary Deployment / Model Rollout

See [Kubernetes](kubernetes.md) for the rolling update this chapter adds model-quality monitoring to.

## What is it?

Canary deployment is a strategy for releasing a new model version to a small fraction of live traffic first, monitoring its real performance, and only rolling it out to all traffic once it's confirmed to behave well — rather than switching every request to the new version all at once.

[Kubernetes](kubernetes.md)'s Deployment gradually replaces old replicas with new ones during a rollout, but that rollout is driven by whether the new pods start successfully, not by whether the new *model* actually performs well on real predictions. Canary deployment adds that missing piece specifically for model changes: route a small percentage of real traffic to the new version, watch its actual prediction quality, and only then decide whether to continue rolling it out or roll back.

## Why does it exist?

[Kubernetes](kubernetes.md) already covered rolling updates for infrastructure changes generally, but a model update carries a different kind of risk than a typical code deployment: a new model version can be technically running perfectly fine — healthy pods, no crashes, fast responses — while making subtly worse predictions than the version it's replacing, a risk Kubernetes' own readiness and liveness probes have no way to catch, since they check whether code is running, not whether its predictions are good. Canary deployment exists specifically to catch that risk: it substitutes model-quality metrics as the gate for whether a rollout continues, while limiting the blast radius to a small percentage of traffic in the meantime.

**Canary deployment vs. a full rollout — a real decision about risk and cost of being wrong.** A full, immediate rollout is reasonable for a low-stakes change already validated thoroughly offline, per [Evaluation](../rag/evaluation.md) and [Agent Evaluation](../agents/agent-evaluation.md)'s own test sets, where speed matters more than caution. A canary rollout is worth the added complexity specifically when a bad version reaching 100% of traffic before anyone notices would be costly — the canary's whole purpose is limiting how much damage a bad model can do before it's caught.

## How does it work?

Deploy the new model version alongside the existing one, routing only a small percentage of live traffic to it while the rest continues to the proven version. Monitor the new version's real prediction quality against a business metric or a proxy for it — the same production monitoring already covered in [Observability](observability.md) — not just infrastructure health. If the new version performs at least as well as the old one over a meaningful sample of real traffic, gradually increase its share toward 100%; if it performs worse, roll back before it affects more users.

## Example

Simulating a genuinely worse replacement model, and checking whether a small canary sample reliably catches it:

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

X, y = make_moons(n_samples=1000, noise=0.3, random_state=0)
X_train, X_live, y_train, y_live = train_test_split(X, y, test_size=0.5, random_state=0)

old_model = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)
new_model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train)  # a real regression

old_accuracy = accuracy_score(y_live, old_model.predict(X_live))
new_accuracy = accuracy_score(y_live, new_model.predict(X_live))
print(old_accuracy, new_accuracy)  # -> 0.896, 0.876 (a genuine, real regression)

correctly_flagged = 0
for seed in range(200):
    rng = np.random.default_rng(seed)
    canary_idx = rng.choice(len(X_live), size=int(0.05 * len(X_live)), replace=False)
    canary_acc = accuracy_score(y_live[canary_idx], new_model.predict(X_live[canary_idx]))
    if canary_acc < old_accuracy:
        correctly_flagged += 1

print(correctly_flagged / 200)  # -> 0.60
```

```text
old model accuracy (full live traffic): 0.896
new model accuracy (full live traffic): 0.876  <- a real, genuine regression

canary sample size: 25 (5% of live traffic)
correctly flagged the regression in 120/200 random canary samples (60.0%)
```

The new model genuinely is worse — a real 2-point accuracy regression on the full traffic. But with only a 5% canary sample (25 examples), the regression is only caught 60% of the time across 200 independent random samples. Forty percent of the time, sampling noise alone makes the worse model look no worse, or even better, on that small a canary slice — exactly the small-sample unreliability [Hypothesis Testing / Confidence Intervals](../statistics/hypothesis-testing-confidence-intervals.md) already demonstrated. A canary percentage isn't automatically a reliable safety net; it needs to be large enough, or observed for long enough, to have real statistical power to catch the regression it's meant to catch.

## Where is it used?

Rolling out a new model version in any production system where a regression reaching all users before detection would be costly — recommendation systems, fraud detection, any model update where "it deployed successfully" isn't the same as "it's actually as good as what it replaced."

## Advantages

- **Limits the blast radius of a bad model version**, catching a genuine regression, as the example shows, before it reaches most of the traffic — when the canary sample is adequate.
- **Uses real production data and real user outcomes** to validate a model change, catching issues offline evaluation might miss entirely.
- **Composes naturally with Kubernetes' own rolling update mechanism**, adding a model-quality gate on top of the infrastructure-level rollout already covered there.

## Limitations

- **A canary sample that's too small has genuinely low statistical power to catch a real regression**, as this chapter's own verified 60% catch rate shows directly — this isn't a hypothetical risk, it's a measured one.
- **Canary deployment adds real time to a rollout** — waiting to accumulate enough canary traffic to trust the comparison delays reaching 100% of users with a good change, not just a bad one.
- **Requires a live, measurable quality signal to compare against**, which isn't always immediately available — some business outcomes take much longer to materialize than a canary window allows.

## Production considerations

- **Canary sample size needs to be chosen based on how large a regression actually needs to be caught**, the same statistical reasoning [Hypothesis Testing / Confidence Intervals](../statistics/hypothesis-testing-confidence-intervals.md) already covered — a smaller regression needs a larger sample to detect reliably.
- **The rollback path needs to be as automated and tested as the rollout path itself** — a canary that correctly detects a regression is useless if rolling back is a manual, slow, or untested process.
- **Canary monitoring needs the same clean-baseline discipline as any anomaly detection**, per [Observability](observability.md) — comparing against a stale or already-degraded "old model" baseline undermines the whole comparison.

## Common mistakes

- **Using a canary percentage too small to reliably catch the regression it's meant to catch**, exactly the risk this chapter's verified 60% catch rate demonstrates concretely.
- **Treating a canary that "looked fine" as conclusive proof of no regression**, rather than checking whether the sample size actually had the statistical power to detect one.
- **Having a rollout path but no equally reliable rollback path**, leaving a detected regression stuck in production anyway.

## Interview questions

### Basic

- What does a canary deployment check that a standard Kubernetes rolling update doesn't?
- Why can a new model version pass all infrastructure health checks and still be worse than what it replaced?

### Intermediate

- Why did the canary sample in this chapter's example fail to catch the regression 40% of the time, despite the regression being real?
- How would you decide what canary traffic percentage is large enough for a given model change?

### Advanced

- Design a canary deployment strategy for a model update where the true business impact takes weeks to materialize, longer than a typical canary window. What would you monitor in the meantime?
- A canary rollout shows no measurable difference from the old model. What would you need to verify before concluding the new model is truly equivalent, rather than the canary sample simply lacking the power to detect a real difference?
