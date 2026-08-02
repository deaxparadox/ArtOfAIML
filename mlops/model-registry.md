# Model Registry

See [Canary Deployment / Model Rollout](canary-deployment.md) for the rollback need this chapter provides the mechanism for.

## What is it?

A model registry is a centralized system for versioning, storing, and tracking the lifecycle of trained models — recording which model version is currently in staging, which is in production, and the full history of versions before it. It's distinct from experiment tracking, which records training runs, by focusing specifically on which trained artifact is deployed where.

[ML Workflow](../machine-learning/ml-workflow.md#production-considerations) named experiment tracking's goal as knowing "which exact snapshot of training data, hyperparameters, and metrics produced a specific model." A model registry is the next step after that: once a trained model artifact exists and has been evaluated, the registry is where it gets a permanent identity — a version — a stage (staging, production, archived), and an audit trail of when and why it moved between those stages.

## Why does it exist?

[ML Workflow](../machine-learning/ml-workflow.md) already flagged the problem at the training-run level: "why did this model behave differently after retraining" needing to be an answerable question. A model registry answers a related but distinct question, specifically about *deployed* models: which exact artifact is currently serving production traffic, which one served it last week, and how do you roll back to a known-good previous version if the current one turns out to have a problem — the exact rollback need [Canary Deployment](canary-deployment.md) already raised. Without a registry, "which model is actually running in production right now" can become a surprisingly hard question to answer with confidence, especially once multiple models and multiple people are involved.

**A model registry vs. a naming convention in cloud storage — a real choice depending on scale.** A small project with one model and one person maintaining it can reasonably track versions with a naming convention and a note in a README. A registry earns its complexity once there are multiple models, multiple environments, or multiple people who need a reliable, queryable answer to "what's currently deployed," rather than relying on institutional knowledge or a document that can go stale.

## How does it work?

Each trained model gets registered with a version identifier, linked back to the exact training run that produced it — the connective tissue back to experiment tracking from [ML Workflow](../machine-learning/ml-workflow.md). The registry tracks which version sits in which stage, and records the history of stage transitions: who promoted what, and when. A deployment system — the [Canary Deployment](canary-deployment.md) process, or a [CI/CD](ci-cd.md) pipeline — queries the registry to know exactly which artifact to deploy, rather than a person needing to remember or guess.

## Example

A working registry, using the exact two model versions and accuracy figures from [Canary Deployment](canary-deployment.md)'s own regression scenario:

```python
class ModelRegistry:
    def __init__(self):
        self.versions = {}
        self.history = []

    def register(self, version, artifact_path, metrics):
        self.versions[version] = {"artifact_path": artifact_path, "metrics": metrics, "stage": "staging"}
        self.history.append(("registered", version))

    def promote(self, version, stage):
        prior_prod = self.get_production()
        for v, info in self.versions.items():
            if info["stage"] == stage:
                info["stage"] = "archived"
        self.versions[version]["stage"] = stage
        self.history.append(("promoted", version, stage, f"previous_prod={prior_prod}"))

    def get_production(self):
        for v, info in self.versions.items():
            if info["stage"] == "production":
                return v
        return None

registry = ModelRegistry()
registry.register("v1", "s3://models/v1.pkl", {"accuracy": 0.896})
registry.register("v2", "s3://models/v2.pkl", {"accuracy": 0.876})  # the regression from Canary Deployment

registry.promote("v1", "production")
print(registry.get_production())  # -> v1

registry.promote("v2", "production")  # promoted after a canary that happened not to catch the regression
print(registry.get_production())  # -> v2

registry.promote("v1", "production")  # rollback
print(registry.get_production())  # -> v1
```

```text
production model: v1
production model after promoting v2: v2
production model after rollback: v1
```

This is the exact scenario [Canary Deployment](canary-deployment.md) verified could happen: v2, the genuinely worse model (0.876 accuracy vs. v1's 0.896), gets promoted to production because that specific canary sample happened not to catch the regression. The registry doesn't prevent that from happening — that's the canary process's job — but it makes the rollback trivial and auditable: one call restores v1 as production, and `registry.history` records exactly when v2 was promoted, when it was rolled back, and what was running before each change.

## Where is it used?

Any team running more than one model version over time, coordinating model deployments across staging and production environments, or needing to answer "what's actually deployed right now, and what was deployed last month" with confidence rather than institutional memory.

## Advantages

- **Makes rollback a fast, reliable operation**, as the example shows directly, instead of a scramble to figure out which artifact was previously in production.
- **Provides a single source of truth for what's currently deployed**, removing the need to trust memory or a possibly-stale document.
- **Creates an audit trail of every promotion and rollback**, useful for both operational debugging and any compliance requirement around model changes.

## Limitations

- **A registry records what was promoted, not why it should have been** — it doesn't replace the evaluation and canary process that should inform a promotion decision in the first place.
- **Adds another system that itself needs to be reliable** — if the registry is wrong or unavailable, deployment decisions built on top of it inherit that uncertainty.
- **Doesn't automatically detect that a promoted model is underperforming** — that's [Observability](observability.md) and [Canary Deployment](canary-deployment.md)'s job; the registry just needs to make acting on that detection fast.

## Production considerations

- **The registry needs to be the actual source of truth a deployment pipeline reads from**, not a record kept alongside a separate, manually-updated deployment process that can drift out of sync with it.
- **Rollback needs to be as fast as promotion**, per [Canary Deployment](canary-deployment.md)'s own point that a slow or untested rollback path undermines the value of catching a regression quickly.
- **Registry access itself needs the same permission discipline as any production system** — who can promote a model to production is a real access-control decision, not an afterthought.

## Common mistakes

- **Tracking deployed model versions informally**, outside any queryable system, until "what's in production" becomes a question only one person can reliably answer.
- **Treating a registry entry as proof a promotion was a good decision**, when it only records that the promotion happened — the actual judgment still comes from evaluation and canary monitoring.
- **Not testing the rollback path until an actual regression forces using it for the first time**, exactly when a slow or broken rollback is most costly.

## Interview questions

### Basic

- What does a model registry track that experiment tracking alone doesn't?
- What is the practical benefit of being able to query "what's in production right now"?

### Intermediate

- Why does a model registry make rollback fast, and why does that matter given what Canary Deployment already showed about regressions slipping through?
- What's the difference between a model registry recording a promotion and that promotion being a good decision?

### Advanced

- Design a model registry's promotion workflow so that a regression caught by canary monitoring triggers an automatic rollback rather than requiring a manual step.
- How would you audit, after the fact, exactly which model version served a specific prediction from three weeks ago?
